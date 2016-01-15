require 'csvigo'

local torch = require 'torch'

captcha = captcha or {}

local caputil = require 'caputil'


local data = torch.class('captcha.data')

function data:__init(dir,loadBool)
  self.dir = dir .. '/'
  self.Yfile = dir .. 'ans.txt'
  --if(loadBool) then self = self:load() end
end

function data:preprocess(dim,Nv)
    return self:loadY()    
        :loadX(dim or 2)   
        :split(Nv or 1000)
        :store()
        :load()
end

function data:loadY()

    local function getHash(Ystr)
        local set = {}
        for i=1,#Ystr do 
            for j=1,string.len(Ystr[i]) do
                local c = string.sub(Ystr[i],j,j)
                set[string.byte(c)] = true
            end
        end
        local hash = {}
        for i=1,500 do
            if(set[i]) then
                table.insert(hash,string.char(i))
            end
        end
        return hash
    end


    local Ystr = caputil.readY(self.Yfile)
    self.N = #Ystr
    self.d = #Ystr[1]
    
    self.hash = getHash(Ystr)
    self.classes = #self.hash
    self.map = caputil.invert(self.hash)
    
    self.Y = torch.zeros(self.N,self.d)
    for i=1,self.N do
        for j=1,self.d do
            local c = string.sub(Ystr[i],j,j)
            self.Y[i][j] = self.map[c]
        end
    end
    return self
end

function data:loadX(dim,start,prefix,suffix)
    self.X = caputil.readImg(self.dir,self.N,start,prefix,suffix,dim)
    self.dims = self.X[1]:size()
    return self
end

function data:loadIdx(idx,dim,start,prefix,suffix)
    self.X = caputil.readImgIdx(self.dir,idx:size(1),start,prefix,suffix,dim,self.X,idx)
    self.dims = self.X[1]:size()
    return self.X
end


function data:randXY(dim, Nt)
    local idx = torch.randperm(self.N)[{{1,Nt}}]:long()
    self.Yt = self.Y:index(1,idx)
    self.Xt = self:loadIdx(idx,dim)
    self.Xv = self.Xt
    self.Yv = self.Yt
end

function data:loadXN(dim)
    local b = 10000
    local j = 1
    for i=1,self.N,b do
        print(i,self.N)
        print(os.date("%X", os.time()))
        collectgarbage()
        print(collectgarbage("count"))
        self.X = caputil.readImg(self.dir,b,i,nil,nil,dim,self.X)
        self.dims = self.X[1]:size()
        torch.save(self.dir .. 'X'..j..'.t7',self.X)
        j = j + 1
        
    end
    self.X = nil
    return self
end

function data:store()
    torch.save(self.dir .. 'data.t7',self)
    return self
end

function data:load()
    return torch.load(self.dir .. 'data.t7')
end


function data:split(Nv)
    local N = self.N
    self.Nv = Nv or N/5
    local I = torch.randperm(N):long()
    local It = I[{{1,N-Nv}}]
    local Iv = I[{{N-Nv+1,N}}]
    self.Xt, self.Yt = self.X:index(1,It),self.Y:index(1,It)
    self.Xv, self.Yv = self.X:index(1,Iv),self.Y:index(1,Iv)
    return self
end



