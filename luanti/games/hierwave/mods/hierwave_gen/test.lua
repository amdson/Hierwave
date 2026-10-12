-- Headless test hooks (main env), all off by default:
--   hierwave_test_radius = R    emerge mapchunks |cx|,|cz| <= R, y -96..31 at startup,
--                               then check random nodes against provider + template
--   hierwave_test_shutdown = true   shut the server down after the test
--   hierwave_ffi_selftest = true    call hw_dummy from luanti/build/libhwdummy.dylib

local S = core.settings
local modpath = core.get_modpath("hierwave_gen")
local function log(level, msg) core.log(level, "[hierwave_gen test] " .. msg) end

if S:get_bool("hierwave_ffi_selftest", false) then
	local ffi = hierwave_gen.ffi  -- from init.lua (insecure env, load time only)
	if not ffi then
		log("action", "FFI selftest: no FFI in the main env")
	else
		local path = modpath .. "/../../../../build/libhwdummy.dylib"
		ffi.cdef("int hw_dummy(int);")
		local ok2, lib = pcall(ffi.load, path)
		if not ok2 then
			log("action", "FFI selftest: ffi.load failed: " .. tostring(lib))
		else
			log("action", "FFI selftest: hw_dummy(20) = " .. lib.hw_dummy(20) .. " (expect 41)")
		end
	end
end

local R = tonumber(S:get("hierwave_test_radius") or "")
if not R then return end
local shutdown = S:get_bool("hierwave_test_shutdown", false)

local function check(p0, p1, nsamples)
	local hg = hierwave_gen
	local gen, provider = hg.gen, hg.provider
	local names = hierwave_nodes.node_names
	local buf = hg.provider_name == "sampler" and nil or {}
	local bad, counts = 0, {}
	-- blocks emerged early may already be unloaded (long city runs)
	core.load_area({x = p0.x, y = -16, z = p0.z}, {x = p1.x, y = 31, z = p1.z})
	local rng = PcgRandom(12345)
	for _ = 1, nsamples do
		local x = rng:next(p0.x, p1.x)
		local y = rng:next(-8, 24)
		local z = rng:next(p0.z, p1.z)
		local r, c = math.floor(-z / 8), math.floor(x / 8)
		local tile
		if hg.provider_name == "sampler" then
			local out = hg.ffi.new("int32_t[1]")
			provider(r, c, 1, 1, out)
			tile = out[0]
		else
			provider(r, c, 1, 1, buf)
			tile = buf[0]
		end
		local want = names[gen.palette_at(x, y, z, tile) + 1]
		local got = core.get_node({x = x, y = y, z = z}).name
		counts[got] = (counts[got] or 0) + 1
		if got ~= want then
			bad = bad + 1
			if bad <= 5 then
				log("error", ("mismatch at (%d,%d,%d) tile %d: got %s, want %s"):format(x, y, z, tile, got, want))
			end
		end
	end
	local parts = {}
	for k, v in pairs(counts) do parts[#parts + 1] = k:gsub("hierwave:", "") .. "=" .. v end
	table.sort(parts)
	return bad, table.concat(parts, " ")
end

core.after(0, function()
	local lo, hi = -32 - 64 * R, 31 + 64 * R
	local p0 = {x = lo, y = -96, z = lo}
	local p1 = {x = hi, y = 31, z = hi}
	local t0 = core.get_us_time()
	log("action", ("emerging %s..%s (%d mapchunks), provider %s, %s env")
		:format(core.pos_to_string(p0), core.pos_to_string(p1),
			(2 * R + 1) ^ 2 * 2, hierwave_gen.provider_name, hierwave_gen.gen_env))
	core.emerge_area(p0, p1, function(blockpos, action, remaining)
		if remaining > 0 then return end
		local dt = (core.get_us_time() - t0) / 1e6
		local nch = (2 * R + 1) ^ 2 * 2
		log("action", ("emerge done: %d mapchunks in %.2f s (%.1f ms/chunk)"):format(nch, dt, dt * 1e3 / nch))
		local st = hierwave_gen.gen.stats
		if st.chunks > 0 then
			log("action", ("main-env generator: %d chunks, tiles %.1f ms/chunk, fill %.1f ms/chunk")
				:format(st.chunks, st.t_tiles * 1e3 / st.chunks, st.t_fill * 1e3 / st.chunks))
		end
		local bad, hist = check(p0, p1, 2000)
		log("action", ("check: %d / 2000 sampled nodes differ from provider+template; sampled: %s"):format(bad, hist))
		log("action", bad == 0 and "TEST PASSED" or "TEST FAILED")
		if shutdown then core.request_shutdown("hierwave test done") end
	end)
end)
