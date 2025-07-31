import julia
julia.install()

from julia.api import Julia
import os 
# Allow compiled modules for fast startup. This is the default.
jl = Julia(compiled_modules=False)

from julia import Pkg
Pkg.activate("/content/SymbolicModelsUtils.jl")
Pkg.instantiate()     # run whenever I update the module not every time 

from julia import SymbolicModelsUtils as smu
from julia import Main

func_list = ["sin", "cos", "sec", "csc", "tan", "cot", "log", "exp", "sqrt", "sinh", "cosh", "sech", "csch", "tanh", "coth", "atan", "asin", "acos", "asinh", "acosh", "atanh", "acoth", "asech", "acsch"]
basic_diadic = ["+", "-", "/", "*", "^","~"]

token_dict = {}
token_dict['[NUM]'] = 1
token_dict.update({chr(i + 96): i+1 for i in range(1, 27)})
token_dict.update({chr(i + 64): i+len(token_dict)+1 for i in range(1, 27)})
token_dict.update({op:i+len(token_dict)+2 for i,op in enumerate(func_list + basic_diadic)})

def make_float(s, return_num=False):
    try:
        s = float(eval(s.replace('//', '/')))
        if return_num:
            return s
        else:
            return 1
    except:
        if return_num:
            return s
        else:
            return token_dict[s]

def classify_operation(token):
    if token==1:
        return 1  # it's a number
    elif 1<token<55:
        return 2  # it's a symbolic variable
    elif 55<=token<61:
        return 3  # it's a trignometric operator
    elif 61<=token<64:
        return 4  # it's a log, exp, sqrt
    elif 64<=token<70:
        return 5  # it's a hyperbolic operator
    elif 70<=token<79:
        return 6  # it's a inverse operator
    elif 79<=token<85:
        return 7  # it's binary operator

def isoperator(token):
    if token<55:
        return 0 # not an operator
    else:
        return 1

def node_feature(node, return_num=False):

    token = make_float(node, return_num)

    return [token, isoperator(token), classify_operation(token)]