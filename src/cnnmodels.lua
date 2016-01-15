local cnnmodels = {}
require 'MultiCrossEntropyCriterion'

function cnnmodels.residual()
    local residual = require 'residual'
    local d,N,K = 16,0,4
    net = nn.Sequential()
    residual.convunit(net,1,d)
    for i=1,K do
        residual.rconvunitN(net,d,N)
        residual.rconvunit2(net,d,true)
        d = 2*d
    end
    local C,H,W = 1,50,200
    local hid,classes,d = 256,24,7
    local y = net:forward(torch.rand(1,C,H,W))
    local N = y:nElement()
    local c,h,w = y:size(2),y:size(3),y:size(4)
    classifier = nn.Sequential()
    classifier:add(nn.SpatialAveragePooling(w,h,1,1))
    classifier:add(nn.View(c))
    classifier:add(nn.Linear(c,hid))
    classifier:add(nn.BatchNormalization(hid))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Linear(hid,classes*d))
    classifier:add(nn.Reshape(d,classes))
    net:add(classifier)
    return net
end


function cnnmodels.residual1()
    local residual = require 'residual'
    local d,N,K = 16,0,4
    net = nn.Sequential()
    residual.convunit(net,1,d)
    for i=1,K do
        residual.rconvunitN(net,d,N)
        residual.rconvunit2(net,d,true)
        d = 2*d
    end
    local C,H,W = 1,20,150
    local hid,classes,d = 256,24,7
    local y = net:forward(torch.rand(1,C,H,W))
    local N = y:nElement()
    local c = N
    classifier = nn.Sequential()
    classifier:add(nn.View(c))
    classifier:add(nn.Linear(c,hid))
    classifier:add(nn.BatchNormalization(hid))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Linear(hid,classes*d))
    classifier:add(nn.Reshape(d,classes))
    net:add(classifier)
    return net
end






function cnnmodels.mct()
    return nn.MultiCrossEntropyCriterion()
end

function cnnmodels.cap(data)
    local tnet = cnnmodels.vggModelLenLite(
    data.dims[1],
    data.dims[2],
    data.dims[3],
    data.d,
    #data.hash)
    local tct = cnnmodels.mct()
    return tnet, tct
end

function cnnmodels.vggModelLen(C,H,W,len,classes)
        local vgg = cnnmodels.vggModel(C,H,W,len * classes)
        vgg:add(nn.Reshape(len,classes))
        return vgg
end
function cnnmodels.vggModelLenLite(C,H,W,len,classes)
        local vgg = cnnmodels.vggModelLite(C,H,W,len * classes)
        vgg:add(nn.Reshape(len,classes))
        return vgg
end

function cnnmodels.vggModel(C,H,W,classes)
    local vgg = nn.Sequential()
    
    local MaxPooling = nn.SpatialMaxPooling
    local function ConvBNReLU(nInputPlane, nOutputPlane, dropout)
      vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
      vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
      vgg:add(nn.ReLU(true))
      if(dropout) then
            vgg:add(nn.Dropout(dropout, nil, true))
      end
      return vgg
    end
    
    ConvBNReLU(C,64,0.3)
    ConvBNReLU(64,64)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(64,128,0.4)
    ConvBNReLU(128,128)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(128,256,0.4)
    ConvBNReLU(256,256,0.4)
    ConvBNReLU(256,256)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(256,512,0.4)
    ConvBNReLU(512,512,0.4)
    ConvBNReLU(512,512)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(512,512,0.4)
    ConvBNReLU(512,512,0.4)
    ConvBNReLU(512,512)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    
    local y = vgg:forward(torch.rand(1,C,H,W))
    local N = y:nElement()
    classifier = nn.Sequential()
    classifier:add(nn.View(N))
    classifier:add(nn.Dropout(0.5,nil,true))
    classifier:add(nn.Linear(N,512))
    classifier:add(nn.BatchNormalization(512))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Dropout(0.5,nil,true))
    classifier:add(nn.Linear(512,classes))
    vgg:add(classifier)
    return vgg
end


function cnnmodels.vggModelLite(C,H,W,classes)
    local vgg = nn.Sequential()
    
    local MaxPooling = nn.SpatialMaxPooling
    local function ConvBNReLU(nInputPlane, nOutputPlane, dropout)
      vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
      vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
      vgg:add(nn.ReLU(true))
      if(dropout) then
            vgg:add(nn.Dropout(dropout, nil, true))
      end
      return vgg
    end
    
    ConvBNReLU(C,64)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(64,128)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(128,256)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(256,256)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(256,256)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(256,256)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    local y = vgg:forward(torch.rand(1,C,H,W))
    local N = y:nElement()
    vgg:add(nn.View(N))
    
    classifier = nn.Sequential()
    classifier:add(nn.Linear(N,512))
    classifier:add(nn.BatchNormalization(512))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Linear(512,classes))
    vgg:add(classifier)
    return vgg
end

return cnnmodels