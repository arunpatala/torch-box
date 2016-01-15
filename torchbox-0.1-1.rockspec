 package = "TorchBox"
 version = "0.1-1"
 source = {
    url = "git://github.com/arunpatala/torch-box",
    tag = "v0.1",
 }
 description = {
    summary = "Torch utilities",
    detailed = [[
    	This will include some utilities written while working with torch
    ]],
    homepage = "http://github.com/arunpatala/torch-box",
    license = "MIT/X11"
 }
 dependencies = {
    "lua >= 5.1",
 }
 build = {
    type = "builtin",
    modules = {
       netutil = "src/netutil.lua",
       captcha = "src/captcha.lua",
       caputil = "src/caputil.lua",
       exp =     "src/exp.lua",
       batch =   "src/batch.lua",
       cnnmodels =   "src/cnnmodels.lua",
       residual =   "src/residual.lua",
       MultiCrossEntropyCriterion =   "src/MultiCrossEntropyCriterion.lua",
    }
 }
