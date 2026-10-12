-- hierwave_nodes: one node per palette entry (luanti/palette.txt), named
-- hierwave:<name>.  Palette id 0 is builtin air.  Flat coloured cubes;
-- water and deep_water are non-spreading liquids (liquid_range = 0), glass
-- and leaves are see-through.  Also sets up a viewer player (fly, fast,
-- noclip; spawn at y = 0).

local modpath = core.get_modpath("hierwave_nodes")
local palette = dofile(modpath .. "/palette.lua")

hierwave_nodes = {
	palette = palette,          -- palette[id + 1] = name
	node_names = {},            -- node_names[id + 1] = "air" | "hierwave:<name>"
}

local function tex(name) return "hierwave_" .. name .. ".png" end

local special = {}

special.glass = {
	drawtype = "glasslike",
	paramtype = "light",
	sunlight_propagates = true,
	use_texture_alpha = "blend",
}

special.leaves = {
	drawtype = "allfaces",
	paramtype = "light",
	sunlight_propagates = true,
	use_texture_alpha = "blend",
	walkable = true,
}

special.window = {
	paramtype = "light",
	light_source = 3,
}

special.lamp = {
	light_source = 14,
}

special.fountain = {
	light_source = 4,
}

local function liquid(name, alpha_colour)
	local flowing = "hierwave:" .. name .. "_flowing"
	local base = {
		drawtype = "liquid",
		tiles = {{name = tex(name), backface_culling = false}},
		special_tiles = {{name = tex(name), backface_culling = false},
			{name = tex(name), backface_culling = true}},
		use_texture_alpha = "blend",
		paramtype = "light",
		walkable = false,
		pointable = true,
		diggable = true,
		buildable_to = true,
		is_ground_content = false,
		drowning = 0,
		liquidtype = "source",
		liquid_alternative_flowing = flowing,
		liquid_alternative_source = "hierwave:" .. name,
		liquid_viscosity = 1,
		liquid_range = 0,          -- never spreads
		liquid_renewable = false,
		post_effect_color = alpha_colour,
		groups = {liquid = 3, oddly_breakable_by_hand = 3},
	}
	return base, flowing
end

for id, name in ipairs(palette) do
	if name == "air" then
		hierwave_nodes.node_names[id] = "air"
	elseif name == "water" or name == "deep_water" then
		local colour = name == "water" and {a = 100, r = 40, g = 100, b = 200}
			or {a = 160, r = 20, g = 50, b = 140}
		local def, flowing = liquid(name, colour)
		def.description = "Hierwave " .. name
		core.register_node(":hierwave:" .. name, def)
		local fdef = table.copy(def)
		fdef.description = "Hierwave " .. name .. " (flowing)"
		fdef.drawtype = "flowingliquid"
		fdef.liquidtype = "flowing"
		fdef.paramtype2 = "flowingliquid"
		fdef.groups = {liquid = 3, not_in_creative_inventory = 1}
		core.register_node(":" .. flowing, fdef)
		hierwave_nodes.node_names[id] = "hierwave:" .. name
	else
		local def = {
			description = "Hierwave " .. name,
			tiles = {tex(name)},
			is_ground_content = false,
			groups = {cracky = 3, oddly_breakable_by_hand = 3},
		}
		for k, v in pairs(special[name] or {}) do def[k] = v end
		core.register_node(":hierwave:" .. name, def)
		hierwave_nodes.node_names[id] = "hierwave:" .. name
	end
end

-- Hand that digs anything instantly (viewing / editing).
core.register_item(":", {
	type = "none",
	wield_image = "blank.png",
	tool_capabilities = {
		full_punch_interval = 0.5,
		max_drop_level = 0,
		groupcaps = {
			cracky = {times = {[1] = 0.2, [2] = 0.2, [3] = 0.2}, uses = 0},
			oddly_breakable_by_hand = {times = {[1] = 0.2, [2] = 0.2, [3] = 0.2}, uses = 0},
			liquid = {times = {[1] = 0.2, [2] = 0.2, [3] = 0.2}, uses = 0},
		},
	},
})

-- Viewer player.
local spawn = {x = 4, y = tonumber(core.settings:get("hierwave_spawn_y") or "") or 0.5, z = -4}
local view_privs = {fly = true, fast = true, noclip = true, interact = true,
	shout = true, teleport = true, debug = true}

core.register_on_joinplayer(function(player)
	local name = player:get_player_name()
	local privs = core.get_player_privs(name)
	local changed = false
	for p in pairs(view_privs) do
		if not privs[p] then privs[p] = true; changed = true end
	end
	if changed then core.set_player_privs(name, privs) end
	player:set_physics_override({speed = 2.0})
	-- the template world is at most 20 blocks tall; see far
	player:set_sky({type = "regular", clouds = false})
end)

core.register_on_newplayer(function(player)
	player:set_pos(spawn)
end)

core.register_on_respawnplayer(function(player)
	player:set_pos(spawn)
	return true
end)
