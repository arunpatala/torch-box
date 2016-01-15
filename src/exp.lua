require 'torch'
require 'optim'
local batch = require 'batch'
box = box or {}

local exp = torch.class('box.exp')

function exp:__init(data,net,ct,batch)
    self.Xt = data.Xt
    self.Yt = data.Yt
    self.Xv = data.Xv
    self.Yv = data.Yv
    self.net = net
    self.ct = ct
    self.batch = batch
    assert(self.batch and self.batch>0)
end

function exp:cuda()
    require 'cunn'
    self.net = self.net:cuda()
    self.ct = self.ct:cuda()
    self.cuda = true
    return self
end

function exp:forward(Xb,Yb)
    self.output = self.net:forward(Xb)
    local loss = self.ct:forward(self.output,Yb)
    return loss,self.output
end


function exp:backward(Xb,Yb)
    local dout = self.ct:backward(self.output,Yb)
    self.net:backward(Xb,dout)
end

function exp:acc(y,Yv)
    local tmp,YYv = y:max(3)
    return 100*YYv:double():squeeze():eq(Yv):double():mean()
end

function exp:accK(y,Yv,k)
    local k = k or Yv:size(2)
    local tmp,YYv = y:max(3)
    return 100*YYv:double():squeeze():eq(Yv):sum(2):eq(k):double():mean()
end

function exp:accuracy(Xv,Yv,k)
    local Xv = Xv or self.Xv
    local Yv = Yv or self.Yv
    self.net:evaluate()
    local y = batch.output(self.net,Xv,self.batch)
    return self:acc(y,Yv),self:accK(y,Yv,k)
end

function exp:sgd(K,sgd_config)
    local x,dx = self.net:getParameters()
    
    print('parameters size ..',#x)
    for k=1,K do
        print(k,K)
        print(os.date("%X", os.time()))
        if(self.epochCallBack) then
            self:epochCallBack()
        end
        local Nt = self.Xt:size(1)
        local lloss = 0
        local YY = nil
        self.net:training()
        for Xb,Yb in batch.iterXY(self.Xt,self.Yt,self.batch) do
            xlua.progress(YY and YY:size(1) or 1, Nt)
            dx:zero()
            local Nb = Xb:size(1)
            local loss,yy = self:forward(Xb,Yb)
            self:backward(Xb,Yb)
            YY = YY and torch.cat(YY,yy,1) or yy
            dx:div(Nb)
            function feval()
                return loss,dx
            end
            local ltmp,tmp = optim.sgd(feval,x,sgd_config)
            lloss = lloss + loss * Nb
        end
        print('loss..'..lloss/Nt)
        --local a,ak = self:accuracy()
        --print('valid .. ',a,ak)
        local a,ak = self:acc(YY,self.Yt)
        print('train .. '.. (a,ak))
        --self:save()
    end
end
local netutil = require 'netutil'
function exp:save()
    torch.save('net.t7',netutil.light(self.net))
end
