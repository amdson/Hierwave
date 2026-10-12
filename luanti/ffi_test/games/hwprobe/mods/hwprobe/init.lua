-- FFI probe: which Lua environments can load a C dylib via LuaJIT FFI?
-- Run: luanti/ffi_test/run_probe.sh
local mp = core.get_modpath("hwprobe")
local dylib = mp .. "/../../../../../build/libhwdummy.dylib"
local function L(s) core.log("action", "PROBE main " .. s) end

L("jit=" .. tostring(jit and jit.version) .. " require=" .. tostring(require))
local ie = core.request_insecure_environment()
L("insecure env=" .. tostring(ie ~= nil))
if ie then
	local ok, ffi = pcall(ie.require, "ffi")
	L("ie.require('ffi') ok=" .. tostring(ok))
	if ok then
		ffi.cdef("int hw_dummy(int);")
		local lib = ffi.load(dylib)
		L("hw_dummy(20)=" .. lib.hw_dummy(20))
	end
end

core.register_mapgen_script(mp .. "/mg.lua")
core.after(0.1, function()
	core.emerge_area({x = 0, y = 0, z = 0}, {x = 1, y = 1, z = 1})
end)
core.after(3, function() core.request_shutdown() end)
