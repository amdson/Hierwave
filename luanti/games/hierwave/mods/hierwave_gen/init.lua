-- hierwave_gen (main env): picks the template file and the tile provider,
-- then generates mapchunks either in the async mapgen env (mapgen.lua) or
-- on the main thread.
--
-- Tile provider:
--   sampler     cache/libhwsampler.dylib + data/<set>.hwm through LuaJIT FFI
--               (needs a Lua env with FFI: the main env of this trusted mod,
--               or any env when mod security is off; and a Luanti binary
--               that may dlopen our dylib, see README.md)
--   placeholder deterministic hash pattern in Lua (common.lua)
-- Where it runs:
--   mapgen      core.register_mapgen_script (emerge threads); used for the
--               placeholder, and for the sampler only if the mapgen env has
--               FFI (mod security off)
--   main        register_on_generated on the main thread; the sampler path
--               with mod security on (FFI via the insecure env)

local modname = core.get_current_modname()
local modpath = core.get_modpath(modname)
local hw = dofile(modpath .. "/common.lua")
local S = core.settings

local function log(level, msg) core.log(level, "[hierwave_gen] " .. msg) end

-- Mapgen: singlenode, chunksize 4 (64^3 mapchunks).
core.set_mapgen_setting("mg_name", "singlenode", true)
core.set_mapgen_setting("chunksize", "4", true)

---------------------------------------------------------------------------
-- settings

local set = S:get("hierwave_set") or "biomes"
local seed = tonumber(S:get("hierwave_seed") or "1") or 1
local provider_mode = S:get("hierwave_provider") or "auto"   -- auto|sampler|placeholder
local env_mode = S:get("hierwave_gen_env") or "auto"         -- auto|main|mapgen
local chunk_tiles = tonumber(S:get("hierwave_chunk_tiles") or "32") or 32
local repo = modpath .. "/../../../../.."
local lib_path = S:get("hierwave_lib") or (repo .. "/cache/libhwsampler.dylib")
local hwm_path = S:get("hierwave_model") or (modpath .. "/data/" .. set .. ".hwm")

---------------------------------------------------------------------------
-- FFI (trusted mod: secure.trusted_mods = hierwave_gen)

local ie = core.request_insecure_environment and core.request_insecure_environment()
local ffi
do
	local req = (ie and ie.require) or require
	local ok, m = pcall(req, "ffi")
	if ok then ffi = m end
end
local security_on = S:get_bool("secure.enable_security", true)
log("action", ("LuaJIT %s; insecure env %s; main-env FFI %s; mod security %s")
	:format(tostring(jit and jit.version), ie and "yes" or "no",
		ffi and "yes" or "no", security_on and "on" or "off"))

local function file_exists(path)
	local open = (ie and ie.io.open) or io.open
	local ok, f = pcall(open, path, "rb")
	if ok and f then f:close(); return true end
	return false
end

---------------------------------------------------------------------------
-- template

local tpl_path = S:get("hierwave_template")
if not tpl_path then
	tpl_path = modpath .. "/data/" .. set .. ".hwt"
	if not file_exists(tpl_path) then
		log("warning", tpl_path .. " missing; using test_data/placeholder.hwt")
		tpl_path = modpath .. "/test_data/placeholder.hwt"
	end
end
local tpl, err = hw.read_hwt(tpl_path)
if not tpl and ie then
	-- paths outside the mod dir need the insecure io
	local saved = io.open
	io.open = ie.io.open
	tpl, err = hw.read_hwt(tpl_path)
	io.open = saved
end
assert(tpl, err)
assert(tpl.N == hw.N, "template N must be 8")
log("action", ("templates %s: n=%d N=%d L=%d"):format(tpl_path, tpl.n, tpl.N, tpl.L))

---------------------------------------------------------------------------
-- provider

local provider, provider_name, new_buffer
local world_addr, model_T

if provider_mode ~= "placeholder" then
	local why
	if not ffi then why = "no FFI in the main env (add hierwave_gen to secure.trusted_mods)"
	elseif not file_exists(lib_path) then why = lib_path .. " missing"
	elseif not file_exists(hwm_path) then why = hwm_path .. " missing"
	end
	if not why then
		local ok, lib = pcall(hw.ffi_lib, ffi, lib_path)
		if not ok then
			why = "ffi.load failed: " .. tostring(lib)
		else
			local model = lib.hw_model_load(hwm_path)
			if model == nil then
				why = "hw_model_load failed on " .. hwm_path
			else
				model_T = lib.hw_model_T(model)
				local world = lib.hw_world_new(model, seed, chunk_tiles)
				provider = hw.sampler_provider(ffi, lib, world)
				new_buffer = function(n) return ffi.new("int32_t[?]", n) end
				world_addr = hw.ptr_to_number(ffi, world)
				provider_name = "sampler"
				log("action", ("sampler: %s, model %s (T=%d, wildcard id T), seed %d, C=%d")
					:format(lib_path, hwm_path, model_T, seed, chunk_tiles))
				if tpl.n < model_T then
					log("warning", ("template file has %d templates < T=%d; out-of-range ids use template 0")
						:format(tpl.n, model_T))
				end
			end
		end
	end
	if why then
		if provider_mode == "sampler" then error("[hierwave_gen] sampler requested but " .. why) end
		log("action", "sampler unavailable (" .. why .. "); using placeholder provider")
	end
end

if not provider then
	provider = hw.placeholder_provider(seed, tpl.n)
	new_buffer = function() return {} end
	provider_name = "placeholder"
end

-- where to generate
local gen_env = env_mode
if gen_env == "auto" then
	gen_env = (provider_name == "sampler" and security_on) and "main" or "mapgen"
end
if gen_env == "mapgen" and provider_name == "sampler" and security_on then
	error("[hierwave_gen] the mapgen env has no FFI with mod security on; use hierwave_gen_env = main")
end
log("action", ("provider %s, set %s, generating in the %s env"):format(provider_name, set, gen_env))

local node_names = hierwave_nodes.node_names
local log_chunks = S:get_bool("hierwave_log_chunks", false)

hierwave_gen = {
	provider = provider,
	provider_name = provider_name,
	template = tpl,
	gen_env = gen_env,
	ffi = ffi,
}

---------------------------------------------------------------------------
-- generation

local gen = hw.new_generator(tpl, node_names, provider, new_buffer, log)
hierwave_gen.gen = gen

if gen_env == "mapgen" then
	core.ipc_set("hierwave_gen:cfg", {
		tpl_path = tpl_path,
		provider = provider_name,
		seed = seed,
		lib_path = lib_path,
		world_addr = world_addr,
		node_names = node_names,
		log_chunks = log_chunks,
	})
	core.register_mapgen_script(modpath .. "/mapgen.lua")
else
	core.register_on_generated(function(minp, maxp, blockseed)
		local vm = core.get_mapgen_object("voxelmanip")
		local ok, dt_tiles, dt_fill = gen.generate(vm, minp, maxp)
		if ok then
			vm:calc_lighting()
			vm:write_to_map()
			if log_chunks then
				log("action", ("chunk %s server env: tiles %.1f ms, fill %.1f ms")
					:format(core.pos_to_string(minp), (dt_tiles or 0) * 1e3, (dt_fill or 0) * 1e3))
			end
		end
	end)
end

dofile(modpath .. "/test.lua")
