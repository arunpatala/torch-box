local caputil = {}



function caputil.readY(path)
    local csv = csvigo.load{path = path, mode = 'raw'}
    local Ystr = {}
    for i=1,#csv do
        table.insert(Ystr,csv[i][1])
    end
    return Ystr
end

function caputil.invert(hash)
        local map = {}
        for k,v in ipairs(hash) do
            map[v] = k
        end
        return map
end

require 'image'

function caputil.readImg(dir,N,start,prefix,suffix,dim,cache)
    local prefix = prefix or ''
    local suffix = suffix or '.png'
    local start = start or 1
    local file = dir .. prefix .. start .. suffix
    local dims = image.load(file):size()
    local X = cache or nil
    if(#dims==2) then 
         X = X or torch.zeros(N,1,dims[1],dims[2])
    else
        if(dim) then
            X = X or torch.zeros(N,1,dims[2],dims[3])
        else
            X = X or torch.zeros(N,dims[1],dims[2],dims[3])
        end
    end
    for i=1,N do
            if(i%1000==0) then print(i, N) end
            if(dim) then
                X[i] = image.load(dir .. prefix .. (start+i-1) .. suffix)[{{dim}}]
            else 
                X[i] = image.load(dir .. prefix .. (start+i-1) .. suffix)
            end
    end
    return X
end

function caputil.readImgIdx(dir,N,start,prefix,suffix,dim,cache,idx)
    local prefix = prefix or ''
    local suffix = suffix or '.png'
    local start = start or 1
    local file = dir .. prefix .. start .. suffix
    local dims = image.load(file):size()
    local X = cache or nil
    if(#dims==2) then 
         X = X or torch.zeros(N,1,dims[1],dims[2])
    else
        if(dim) then
            X = X or torch.zeros(N,1,dims[2],dims[3])
        else
            X = X or torch.zeros(N,dims[1],dims[2],dims[3])
        end
    end
    
    for i=1,N do
            if(i%1000==0) then print(i, N) end
            local j = idx[i]
            if(dim) then
                X[i] = image.load(dir .. prefix .. (j) .. suffix)[{{dim}}]
            else 
                X[i] = image.load(dir .. prefix .. (j) .. suffix)
            end
    end
    return X
end

function caputil.split(X,Y,Nv)
    local N = X:size(1)
    local Nv = Nv or self.N/5
    local I = torch.randperm(N):long()
    local It = I[{{1,N-Nv}}]
    local Iv = I[{{N-Nv+1,N}}]
    local Xt, Yt = X:index(1,It),Y:index(1,It)
    local Xv, Yv = X:index(1,Iv),Y:index(1,Iv)
    return Xt,Yt,Xv,Yv
end



return caputil

