local netutil = {}

local bna = require('BN-absorber')

function netutil.lightSave(f,net)
	torch.save(f,netutil.light(net))
end

function netutil.light(net)
	net = bna(net)
	for i=1,#net.modules do
        	local m = net.modules[i]
            if(m.modules) then netutil.light(m) end
	        if(m.output) then m.output = torch.CudaTensor() end
        	if(m.gradInput) then m.gradInput = torch.CudaTensor()  end
	        if(m.gradWeight) then m.gradWeight = torch.CudaTensor()  end
        	if(m.gradBias) then m.gradBias = torch.CudaTensor()  end
	        if(m.fgradInput) then m.fgradInput = torch.CudaTensor()  end
	end
	return net
end

function netutil.bna(net)
    bna(net)
    local mods = net:findModules('nn.Sequential')
    for i=1,#mods do
        bna(mods[i])
    end
    return net
end


return netutil
