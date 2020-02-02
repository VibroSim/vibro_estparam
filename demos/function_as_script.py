# function_as_script.py:
# Execute a Python function as if it were part of a script.

# How to use:
#  * Keep variable names in script EXACTLY matching
#    variable names in function.
#  * Assign parameters in script
#  *   from function_as_script import scriptify
#  *   scriptified_function=scriptify(my_function)
#  * Call scriptified_function with parameters
#  * Can supply keyword arguments to cause assignments
#    from globals of different names. The assignments
#    occur with each script call in the __main__
#    namespace
#  * NOTE: If used within LIMATIX ProcessTrak  instead 
#    of assigning into the __main__ namespace it will 
#    instead assign into the globals of the current
#    processtrak step.  
#  * Default arguments work, but relying on them is
#    dangerous if you have calls with/without the
#    default arguments in the script, because
#    the values assigned in one call will be
#    reused in the next call (because they are
#    now in the __main__ namespace) 
#  * Note that local variables assigned in the
#    function WILL OVERRIDE VARIABLES IN THE
#    __main__ NAMESPACE! These changes will
#    persiste even after the function returns!!!
#  * After calling scriptify() all global variables
#    from the function's module will be mapped into
#    the __main__ namespace. scriptify() will
#    fail if there is a namespace conflict.
#
# WARNING: For the moment, if the function depends on
# __future__ statements, those may cause the
# scriptification to fail!

import sys
import inspect
import ast

def add_to_lineos(astobj,lineno_increment):
    # (NO_LONGER_NEEDED!)
    # Increment this object
    if hasattr(astobj,"lineno"):
        astobj.lineno += lineno_increment
        pass

    # recursively increment linenumbers for all fields
    for field in astobj._fields:
        if hasattr(astobj,field):
            field_data = getattr(astobj,field)

            # if the field is a list
            if isinstance(field_data,list):
                # Operate on each elemetn of list
                [ add_to_linenos(field_datum) for field_datum in field_data ]
                pass
            else:
                # otherwise the field should be an AST object
                add_to_linenos(field_data)
                pass
            
            pass
        pass
    pass

def py275_exec_bug_workaround(codeobj,globs,locs):
    """Workaround for bug in Python 2.7.5 where if we exec
directly inside a nested function we get an error"""

    exec(codeobj,globs,locs)
    pass


def scriptify(callable):

    codeobj = callable.__code__

    #(sourcelines,firstlineno) = inspect.getsourcelines(callable)
    #sourcecode = "".join(sourcelines)

    sourcefile=inspect.getsourcefile(callable)
    sourcecode=open(sourcefile,"r").read()
    

    syntree = ast.parse(sourcecode)
    assert(syntree.__class__ is ast.Module)

    # Find the function/method definition
    synsubtree=None
    if callable.__class__.__name__=="method":
        # Got a method not a class
        if callable.__self__.__class__ is type:
            # This is a classmethod we got
            classname=callable.__self__.__name__
            pass
        else:
            # This is a regular (object) method
            classname = callable.__self__.__class__.__name__
            pass
        
        # Find the correct method definition
        for BodyEl in syntree.body:
            if isinstance(BodyEl,ast.ClassDef):
                # Found a function definition
                if BodyEl.name==classname:
                    # Found our class
                    # seek out the method

                    for SubBodyEl in BodyEl.body:
                        if isinstance(SubBodyEl,ast.FunctionDef):
                            # Found a function definition
                            if SubBodyEl.name==callable.__name__:
                                # Found our function
                                
                                synsubtree = SubBodyEl
                                break
                            pass
                        pass
                    
                    break
                pass
            pass
        pass
    else:
        # Find the correct function definition
        
        for BodyEl in syntree.body:
            if isinstance(BodyEl,ast.FunctionDef):
                # Found a function definition
                if BodyEl.name==callable.__name__:
                    # Found our function
                    synsubtree = BodyEl
                    break
                pass
            pass
        pass
    if synsubtree is None:
        raise ValueError("Definition of function/method %s not found in %s" % (callable.__name__,sourcefile))
    
    
    ## Iterate over syntree, correcting line numbers
    #add_to_linenos(syntree,firstlineno-1)

    #assert(syntree.body[0].__class__ is ast.FunctionDef)

    args = synsubtree.args.args
    argnames = [ arg.id if hasattr(arg,"id") else arg.arg for arg in args ]
    
    # Extract default arguments
    assert((len(synsubtree.args.defaults) == 0 and callable.__defaults__ is None) or len(synsubtree.args.defaults) == len(callable.__defaults__))

    
    mandatoryargnames = argnames[:(len(argnames)-len(synsubtree.args.defaults))]
    defaultargnames = argnames[(len(argnames)-len(synsubtree.args.defaults)):]


    # Search for a context to write to. Normally this would be
    # the __main__ globals. But we check first whether there is 
    # a processtrak step. Do this by inspecting the stack:
    context = None
    try: 
        stackframe=None
        stacktrace = inspect.stack()
        for stackcnt in range(1,len(stacktrace)):
            stackframe=stacktrace[stackcnt][0]
            stack_global_dict = dict(inspect.getmembers(stackframe))["f_globals"]
            if "__processtrak_stepname" in stack_global_dict:
                # This is a global dictionary created by processtrak 
                # for a processtrak step. 
                # ... use this context!
                context = stack_global_dict
                break
            pass
        pass
    finally:
        # Avoid keeping references around
        del stackframe
        del stacktrace
        pass
        
    if context is None:
        # Did not find a processtrak context
        # Use __main__ global dictionary
        context = sys.modules["__main__"].__dict__
        pass
        

    CodeModule=ast.Module(body=synsubtree.body,lineno=0)

    # if CodeModule ends with return statement, replace with
    # assignment to _fas_returnval
    has_return=False
    if isinstance(CodeModule.body[-1],ast.Return) and CodeModule.body[-1].value is not None:
        has_return=True
        new_assignment = ast.Assign(targets=[ast.Name(id="_fas_returnval",lineno=CodeModule.body[-1].lineno,col_offset=0,ctx=ast.Store())],value=CodeModule.body[-1].value,lineno=CodeModule.body[-1].lineno,col_offset=0)
        CodeModule.body.pop()
        CodeModule.body.append(new_assignment)
        pass
    
    

    codeobj = compile(CodeModule,inspect.getsourcefile(callable),"exec",dont_inherit=True)  # should be able to set flags based on __future__ statments in original source module, but we don't currently do this
    
    def scriptified(*args,**kwargs):
        # Pass arguments
        if callable.__class__.__name__=="method":
            argshift=1
            pass
        else:
            argshift=0
            pass
        
        for argcnt in range(len(mandatoryargnames)):
            argname=mandatoryargnames[argcnt]
            if argcnt==0 and callable.__class__.__name__=="method": # ("self" parameter)
                argvalue=callable.__self__
                pass
            elif argcnt-argshift < len(args):
                argvalue = args[argcnt-argshift]
                pass
            elif argname in kwargs:
                argvalue=kwargs[argname]
                pass
            else:
                raise ValueError("Argument %s must be provided" % (argname))

            context[argname] = argvalue
            pass

        # Optional arguments
        for argcnt in range(len(defaultargnames)):
            argname=defaultargnames[argcnt]
            if argcnt+len(mandatoryargnames)-argshift < len(args):
                argvalue = args[argcnt+len(mandatoryargnames)-argshift]
                pass
            elif argname in kwargs:
                argvalue=kwargs[argname]
                pass
            else:
                # default value
                argvalue=callable.__defaults__[argcnt]
                pass
                
            context[argname] = argvalue
            pass
        
 
        #
        ## Assign default arguments where needed        
        #for defaultargnum in range(len(defaultargnames)):
        #    argname=defaultargnames[defaultargnum]
        #    if not argname in context and not argname in kwargs:
        #        context[argname] = callable.__defaults__[defaultargnum]
        #        pass
        #    pass
        
        # execute!        
        #exec(codeobj,context,context)

        py275_exec_bug_workaround(codeobj,context,context)
        if has_return:
            retval=context["_fas_returnval"]
            pass
        else:
            retval=None
        return retval
    

    # Add globals from function module into __main__ namespace
    if hasattr(callable,"__globals__"):
        globaldict=callable.__globals__
        pass
    else:
        globaldict=sys.modules[callable.__module__].__dict__
        pass

    for varname in globaldict:
        if varname.startswith("__"):
            # do not transfer any variables starting with "__"
            pass
        elif varname in context:
            if hasattr(context[varname],"__name__") and context[varname].__name__=="scriptified":
                # Do not object to (but do not overwrite)
                # a scriptified function
                pass
            elif context[varname] is not globaldict[varname]:
                raise ValueError("Variable conflict between __main__.%s and %s.%s" % (varname,callable.__module__,varname))
            pass
        else:
            # assign variable
            context[varname]=globaldict[varname]
            pass
        pass
    
    
    return scriptified

    
