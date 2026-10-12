-- hierwave_gen, async mapgen env (one Lua state per emerge thread).
-- Config comes from the main env through core.ipc_get.

local modpath = core.get_modpath("hierwave_gen")
local hw = dofile(modpath .. "/common.lua")

local function log(level, msg) core.log(level, "[hierwave_gen mapgen] " .. msg) end

local cfg = assert(core.ipc_get("hierwave_gen:cfg"), "hierwave_gen: no config from the main env")
local tpl = assert(hw.read_hwt(cfg.tpl_path))

local provider, new_buffer
if cfg.provider == "sampler" then
	-- only reachable with mod security off (require is disabled otherwise)
	local ffi = require("ffi")
	local lib = hw.ffi_lib(ffi, cfg.lib_path)  -- same dlopen handle as the main env
	local world = hw.number_to_ptr(ffi, "hw_world*", cfg.world_addr)
	provider = hw.sampler_provider(ffi, lib, world)
	new_buffer = function(n) return ffi.new("int32_t[?]", n) end
else
	provider = hw.placeholder_provider(cfg.seed, tpl.n, cfg.layers)
	new_buffer = function() return {} end
end

local gen = hw.new_generator(tpl, cfg.node_names, provider, new_buffer, log, cfg.layers)

core.register_on_generated(function(vm, minp, maxp, blockseed)
	local ok, dt_tiles, dt_fill = gen.generate(vm, minp, maxp)
	if ok then
		vm:calc_lighting()
		if cfg.log_chunks then
			log("action", ("chunk %s: tiles %.1f ms, fill %.1f ms")
				:format(core.pos_to_string(minp), (dt_tiles or 0) * 1e3, (dt_fill or 0) * 1e3))
		end
	end
end)
