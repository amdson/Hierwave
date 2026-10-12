-- Shared by the main env (init.lua) and the mapgen env (mapgen.lua):
-- template file reader, tile providers, and the per-mapchunk fill.
--
-- Contract (notes/luanti_plan.md):
--   tile (r, c): r grows south, c east; N = 8 blocks per tile side.
--   world x = 8 c + j, world z = -(8 r + i); template layer k is world y = k - 4.
--   below layer 0: stone; above layer L - 1: air.

local hw = {}

hw.N = 8
hw.Y0 = -4  -- world y of template layer 0

local floor = math.floor
local byte = string.byte

-- wall clock seconds (os.clock is process CPU time, wrong across threads)
local function now()
	return core.get_us_time() / 1e6
end
hw.now = now

---------------------------------------------------------------------------
-- .hwt template file: "HWT1", u32 n, u32 N, u32 L, u8 data[n][L][N][N]

local function u32(s, o)
	local a, b, c, d = byte(s, o, o + 3)
	return a + 256 * (b + 256 * (c + 256 * d))
end

function hw.read_hwt(path)
	local f, err = io.open(path, "rb")
	if not f then return nil, "cannot open " .. path .. ": " .. tostring(err) end
	local s = f:read("*a")
	f:close()
	if #s < 16 or s:sub(1, 4) ~= "HWT1" then
		return nil, path .. ": not an HWT1 file"
	end
	local n, N, L = u32(s, 5), u32(s, 9), u32(s, 13)
	if #s ~= 16 + n * L * N * N then
		return nil, ("%s: size %d, expected %d"):format(path, #s, 16 + n * L * N * N)
	end
	return {n = n, N = N, L = L, data = s, path = path}
end

---------------------------------------------------------------------------
-- Placeholder tile provider: deterministic hash pattern, ids in [0, n).
-- Coarse 4x4-tile blocks pick a "theme" so the result has some structure.

local bit = bit

local function hash3(a, b, c)
	local h = bit.tobit(0x9E3779B9)
	h = bit.bxor(h, bit.tobit(a * 0x85EBCA6B)); h = bit.rol(h, 13); h = bit.tobit(h * 5 + 0xE6546B64)
	h = bit.bxor(h, bit.tobit(b * 0xC2B2AE35)); h = bit.rol(h, 13); h = bit.tobit(h * 5 + 0xE6546B64)
	h = bit.bxor(h, bit.tobit(c * 0x27D4EB2F)); h = bit.rol(h, 13); h = bit.tobit(h * 5 + 0xE6546B64)
	h = bit.bxor(h, bit.rshift(h, 16)); h = bit.tobit(h * 0x85EBCA6B)
	h = bit.bxor(h, bit.rshift(h, 13))
	return bit.band(h, 0x7FFFFFFF)
end

-- themes over the placeholder templates (luanti/tools/make_placeholder_hwt.py):
-- 0 deep water, 1 water, 2 sand, 3 grass, 4 tree, 5 house, 6 road, 7 tower
local THEMES = {
	{0, 0, 0, 1},        -- sea
	{1, 2, 2, 3},        -- shore
	{3, 3, 4, 4, 3},     -- park
	{3, 5, 5, 6, 4},     -- village
	{6, 7, 7, 5, 6},     -- town
}

function hw.placeholder_provider(seed, n)
	return function(r0, c0, H, W, out)
		for a = 0, H - 1 do
			local r = r0 + a
			for b = 0, W - 1 do
				local c = c0 + b
				local th = THEMES[hash3(floor(r / 4), floor(c / 4), seed) % #THEMES + 1]
				local t = th[hash3(r, c, seed + 1) % #th + 1]
				out[a * W + b] = t % n
			end
		end
		return 0
	end
end

---------------------------------------------------------------------------
-- Sampler tile provider through LuaJIT FFI (contract C API).

local CDEF = [[
typedef struct hw_model hw_model;
typedef struct hw_world hw_world;
hw_model* hw_model_load(const char* path);
void      hw_model_free(hw_model*);
int       hw_model_T(const hw_model*);
int       hw_tiles(const hw_model*, uint64_t seed, int64_t y0, int64_t x0,
                   int H, int W, int32_t* out);
hw_world* hw_world_new(const hw_model*, uint64_t seed, int C);
int       hw_world_region(hw_world*, int64_t y0, int64_t x0, int H, int W, int32_t* out);
void      hw_world_free(hw_world*);
]]

local cdef_done = {}

function hw.ffi_lib(ffi, lib_path)
	if not cdef_done[ffi] then
		ffi.cdef(CDEF)
		cdef_done[ffi] = true
	end
	return ffi.load(lib_path)
end

-- provider over an existing hw_world pointer; out is an int32_t[] cdata
function hw.sampler_provider(ffi, lib, world)
	return function(r0, c0, H, W, out)
		return lib.hw_world_region(world, r0, c0, H, W, out)
	end
end

function hw.ptr_to_number(ffi, p)
	return tonumber(ffi.cast("uintptr_t", p))
end

function hw.number_to_ptr(ffi, ctype, n)
	return ffi.cast(ctype, ffi.cast("uintptr_t", n))
end

---------------------------------------------------------------------------
-- Generator: fills mapchunk [minp, maxp] of a VoxelManip.

-- node_names[id + 1] = node name for palette id
function hw.new_generator(tpl, node_names, provider, new_buffer, log)
	local N, L = tpl.N, tpl.L
	local NN = N * N
	local LNN = L * NN
	local ntpl = tpl.n
	local s = tpl.data
	local Y0 = hw.Y0
	local Ytop = Y0 + L - 1

	local cid = {}
	for id, name in ipairs(node_names) do
		cid[id - 1] = core.get_content_id(name)
	end
	local c_stone = core.get_content_id("hierwave:stone")
	local c_air = core.get_content_id("air")
	-- palette ids in the template string -> content ids
	local pal = {}
	for id = 0, 255 do pal[id] = cid[id] or c_air end

	local data = {}
	local tiles, tiles_len = nil, 0
	local bad_ids = 0

	local gen = {stats = {chunks = 0, t_tiles = 0, t_fill = 0, bad_ids = 0}}

	-- tile region for a chunk: rows r0..r1, cols c0..c1
	function gen.tile_region(minp, maxp)
		local c0, c1 = floor(minp.x / N), floor(maxp.x / N)
		local r0, r1 = floor(-maxp.z / N), floor(-minp.z / N)
		return r0, c0, r1 - r0 + 1, c1 - c0 + 1
	end

	-- palette id at a world position, given tile ids (for checks)
	function gen.palette_at(x, y, z, tile)
		if y < Y0 then return 1 end  -- stone
		if y > Ytop then return 0 end
		local zz = -z
		local i, j = zz % N, x % N
		if tile < 0 or tile >= ntpl then tile = 0 end
		return byte(s, 17 + tile * LNN + (y - Y0) * NN + i * N + j)
	end

	function gen.generate(vm, minp, maxp)
		if minp.y > Ytop then return false end
		local t0 = now()
		local emin, emax = vm:get_emerged_area()
		local area = VoxelArea(emin, emax)
		vm:get_data(data)
		local ystride = area.ystride
		local zstride = area.zstride

		if maxp.y < Y0 then
			for z = minp.z, maxp.z do
				for y = minp.y, maxp.y do
					local vi = area:index(minp.x, y, z)
					for _ = minp.x, maxp.x do
						data[vi] = c_stone
						vi = vi + 1
					end
				end
			end
			vm:set_data(data)
			gen.stats.chunks = gen.stats.chunks + 1
			local dt = now() - t0
			gen.stats.t_fill = gen.stats.t_fill + dt
			return true, 0, dt
		end

		local r0, c0, H, W = gen.tile_region(minp, maxp)
		if H * W > tiles_len then
			tiles = new_buffer(H * W)
			tiles_len = H * W
		end
		local rc = provider(r0, c0, H, W, tiles)
		if rc ~= 0 then
			log("error", ("tile provider failed (%s) at r0=%d c0=%d"):format(tostring(rc), r0, c0))
			return false
		end
		local t1 = now()

		local ylo = minp.y
		local yhi = maxp.y < Ytop and maxp.y or Ytop
		for z = minp.z, maxp.z do
			local zz = -z
			local r = floor(zz / N)
			local i = zz - r * N
			local trow = (r - r0) * W
			for y = ylo, yhi do
				local vi = area:index(minp.x, y, z)
				if y < Y0 then
					for _ = minp.x, maxp.x do
						data[vi] = c_stone
						vi = vi + 1
					end
				else
					local koff = 17 + (y - Y0) * NN + i * N
					for x = minp.x, maxp.x do
						local c = floor(x / N)
						local t = tiles[trow + c - c0]
						if t < 0 or t >= ntpl then
							bad_ids = bad_ids + 1
							t = 0
						end
						data[vi] = pal[byte(s, t * LNN + koff + (x - c * N))]
						vi = vi + 1
					end
				end
			end
		end
		vm:set_data(data)
		local t2 = now()
		local st = gen.stats
		st.chunks = st.chunks + 1
		st.t_tiles = st.t_tiles + (t1 - t0)
		st.t_fill = st.t_fill + (t2 - t1)
		st.bad_ids = bad_ids
		return true, t1 - t0, t2 - t1
	end

	return gen
end

return hw
