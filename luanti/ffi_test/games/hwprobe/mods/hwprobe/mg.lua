-- runs in the mapgen (emerge) environment
local function L(s) core.log("action", "PROBE mg " .. s) end
L("jit=" .. tostring(jit and jit.version)
	.. " require=" .. tostring(require)
	.. " request_insecure_environment=" .. tostring(core.request_insecure_environment)
	.. " get_modpath=" .. tostring(core.get_modpath)
	.. " io.open=" .. tostring(io and io.open)
	.. " ipc_get=" .. tostring(core.ipc_get))
if core.get_modpath then L("modpath=" .. tostring(core.get_modpath("hwprobe"))) end
if core.request_insecure_environment then
	local ok, ie = pcall(core.request_insecure_environment)
	L("insecure env ok=" .. tostring(ok) .. " " .. tostring(ie))
	if ok and ie then
		local ok2, ffi = pcall(ie.require, "ffi")
		L("ie.require('ffi') ok=" .. tostring(ok2))
		if ok2 then
			ffi.cdef("int hw_dummy(int);")
			local lib = ffi.load(core.get_modpath("hwprobe") .. "/../../../../../build/libhwdummy.dylib")
			L("hw_dummy(20)=" .. lib.hw_dummy(20))
		end
	end
end
if require then
	local ok, ffi = pcall(require, "ffi")
	L("plain require('ffi') ok=" .. tostring(ok) .. " " .. tostring(ffi))
end
core.register_on_generated(function(vm, minp, maxp, seed)
	L("on_generated " .. core.pos_to_string(minp) .. " " .. core.pos_to_string(maxp))
end)
