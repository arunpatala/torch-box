local batch = {}

function batch.output(net,Xt,b,cuda)
    local Yt = nil 
    for Xb in batch.iterX(Xt,b,cuda) do
        local Yb = net:forward(Xb)
        Yt = Yt and torch.cat(Yt,Yb,1) or Yb
    end
    return Yt
end

function batch.iterX(Xt,b,cuda)
    local cuda = cuda or true
    local Nt = Xt:size(1)
    local i = 1
    local function biter()
        if(i>Nt) then return nil end
        local j = math.min(i+b-1,Nt)
        local Xb = Xt[{{i,j}}]
        if(cuda) then
            require 'cunn'
            Xb = Xb:cuda()
        end
        i = j + 1
        return Xb
    end
    return biter
end

function batch.zipIter(it1,it2)
    local function ziter()
        local X = it1()
        local Y = it2()
        return X,Y
    end
    return ziter
end

function batch.fIter(it1,f)
    local function ziter()
        local X = it1()
        return f(X)
    end
    return ziter
end

function batch.iterXY(Xt,Yt,b,cuda)
    return batch.zipIter(batch.iterX(Xt,b,cuda),batch.iterX(Yt,b,cuda))
end

return batch