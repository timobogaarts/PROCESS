# Stage check: non-anticipativity of claimed build outputs

Graph: as measured (156, duct root find kept), 156 nodes; claims: `kinds.CLAIMED_BUILD_OUTPUTS`, sections 1a-1d (the handoff's) (25).

## Summary

| belief set | leaves | nodes 1st | nodes 2nd | nodes recourse | owned 1st | owned 2nd | 1st-stage frac | violations |
|---|---|---|---|---|---|---|---|---|
| all | 146 + 16 | 27 | 112 | 17 | 92 | 395 | 0.17 | 13 |
| sampled | 26 + 2 | 36 | 120 | 0 | 103 | 441 | 0.23 | 13 |
| build | 22 + 2 | 60 | 96 | 0 | 191 | 353 | 0.38 | 4 |

## Violations -- belief set 'all' (13)

| place | section | producing node | stage | # leaves | leaves |
|---|---|---|---|---|---|
| `.heat_transport.n_primary_heat_exchangers` | 1c | `.power.component_thermal_powers` | second | 52 | .costs.abktflnc, .current_drive.eta_ecrh_injector_wall_plug, .current_drive.p_hcd_primary_extra_heat_mw, .divertor.tdiv, .divertor.xpertin, .fwbs.declblkt, .fwbs.declfw, .fwbs.declshld +44 more |
| `.fwbs.life_fw_fpy` | 1b | `.stellarator.fwbs.fw_blanket_shield_geometry` | second | 41 | .costs.abktflnc, .divertor.tdiv, .divertor.xpertin, .fwbs.fvolsi, .fwbs.fvolso, .impurity_radiation.f_nd_impurity_electron_array[10], .impurity_radiation.f_nd_impurity_electron_array[11], .impurity_radiation.f_nd_impurity_electron_array[12] +33 more |
| `.vacuum.dia_vv_vacuum_ducts` | 1b | `.vacuum.vacuum_old` | second | 35 | .impurity_radiation.f_nd_impurity_electron_array[10], .impurity_radiation.f_nd_impurity_electron_array[11], .impurity_radiation.f_nd_impurity_electron_array[12], .impurity_radiation.f_nd_impurity_electron_array[13], .impurity_radiation.f_nd_impurity_electron_array[2], .impurity_radiation.f_nd_impurity_electron_array[3], .impurity_radiation.f_nd_impurity_electron_array[4], .impurity_radiation.f_nd_impurity_electron_array[5] +27 more |
| `.vacuum.n_vac_pumps_high` | 1b | `.vacuum.vacuum_old` | second | 35 | .impurity_radiation.f_nd_impurity_electron_array[10], .impurity_radiation.f_nd_impurity_electron_array[11], .impurity_radiation.f_nd_impurity_electron_array[12], .impurity_radiation.f_nd_impurity_electron_array[13], .impurity_radiation.f_nd_impurity_electron_array[2], .impurity_radiation.f_nd_impurity_electron_array[3], .impurity_radiation.f_nd_impurity_electron_array[4], .impurity_radiation.f_nd_impurity_electron_array[5] +27 more |
| `.build.dr_fw_inboard` | 1b | `.stellarator.build` | recourse | 1 | .tfcoil.tftmp |
| `.build.dr_fw_outboard` | 1b | `.stellarator.build` | recourse | 1 | .tfcoil.tftmp |
| `.build.dr_shld_vv_gap_outboard` | 1b | `.stellarator.build` | recourse | 1 | .tfcoil.tftmp |
| `.build.dr_tf_outboard` | 1b | `.stellarator.build` | recourse | 1 | .tfcoil.tftmp |
| `.stellarator.wp_width_r_min` | 1a | `^problem.stellarator.coils.intersect` | recourse | 1 | .tfcoil.tftmp |
| `.tfcoil.dr_tf_wp_with_insulation` | 1a | `.stellarator.coils.winding_pack_total_size_post` | recourse | 1 | .tfcoil.tftmp |
| `.tfcoil.dx_tf_wp_primary_toroidal` | 1a | `.stellarator.coils.winding_pack_total_size_post` | recourse | 1 | .tfcoil.tftmp |
| `.tfcoil.dx_tf_wp_secondary_toroidal` | 1a | `.stellarator.coils.winding_pack_total_size_post` | recourse | 1 | .tfcoil.tftmp |
| `.tfcoil.j_tf_wp` | 1a | `.stellarator.coils.winding_pack_total_size_post` | recourse | 1 | .tfcoil.tftmp |

## Confirmed first-stage -- belief set 'all' (12)

| place | section | producing node |
|---|---|---|
| `.build.dr_cs` | 1d | `.initialisation.stellarator_solenoid_absent` |
| `.build.dr_cs_tf_gap` | 1d | `.initialisation.stellarator_solenoid_absent` |
| `.buildings.esbldgm3` | 1d | `.initialisation.energy_storage_building_volume` |
| `.physics.aspect` | 1a | `.stellarator.default_aspect_ratio` |
| `.stellarator.coilcurrent` | 1a | `.stellarator.coils.coil_current` |
| `.stellarator.f_st_i_total` | 1a | `.stellarator.coils.coil_current` |
| `.stellarator.r_coil_major` | 1a | `.stellarator.stellarator_scaling_factors` |
| `.stellarator.r_coil_minor` | 1a | `.stellarator.stellarator_scaling_factors` |
| `.tfcoil.dr_tf_plasma_case` | 1a | `.stellarator.coils.coil_casing` |
| `.tfcoil.dx_tf_side_case_min` | 1a | `.stellarator.coils.coil_casing` |
| `.tfcoil.n_tf_coils` | 1a | `.stellarator.stellarator_scaling_factors` |
| `.vacuum.d_duct` | 1b | `^problem.vacuum.duct_diameter_root_find` |

## Violations -- belief set 'sampled' (13)

| place | section | producing node | stage | # leaves | leaves |
|---|---|---|---|---|---|
| `.heat_transport.n_primary_heat_exchangers` | 1c | `.power.component_thermal_powers` | second | 22 | .constraints.f_j_tf_wp_critical_max, .costs.life_plant, .current_drive.eta_ecrh_injector_wall_plug, .divertor.tdiv, .fwbs.declblkt, .fwbs.declfw, .fwbs.dr_fw_wall, .fwbs.f_p_blkt_multiplication +14 more |
| `.fwbs.life_fw_fpy` | 1b | `.stellarator.fwbs.fw_blanket_shield_geometry` | second | 17 | .constraints.f_j_tf_wp_critical_max, .costs.life_plant, .divertor.tdiv, .fwbs.dr_fw_wall, .fwbs.fhole, .impurity_radiation.f_nd_impurity_electron_array[13], .physics.alphan, .physics.alphat +9 more |
| `.vacuum.dia_vv_vacuum_ducts` | 1b | `.vacuum.vacuum_old` | second | 12 | .constraints.f_j_tf_wp_critical_max, .fwbs.dr_fw_wall, .impurity_radiation.f_nd_impurity_electron_array[13], .physics.alphan, .physics.alphat, .physics.f_nd_alpha_thermal_electron, .physics.f_p_alpha_plasma_deposited, .physics.f_sync_reflect +4 more |
| `.vacuum.n_vac_pumps_high` | 1b | `.vacuum.vacuum_old` | second | 12 | .constraints.f_j_tf_wp_critical_max, .fwbs.dr_fw_wall, .impurity_radiation.f_nd_impurity_electron_array[13], .physics.alphan, .physics.alphat, .physics.f_nd_alpha_thermal_electron, .physics.f_p_alpha_plasma_deposited, .physics.f_sync_reflect +4 more |
| `.build.dr_fw_inboard` | 1b | `.stellarator.build` | second | 3 | .constraints.f_j_tf_wp_critical_max, .fwbs.dr_fw_wall, .tfcoil.dx_tf_wp_insulation |
| `.build.dr_fw_outboard` | 1b | `.stellarator.build` | second | 3 | .constraints.f_j_tf_wp_critical_max, .fwbs.dr_fw_wall, .tfcoil.dx_tf_wp_insulation |
| `.build.dr_shld_vv_gap_outboard` | 1b | `.stellarator.build` | second | 3 | .constraints.f_j_tf_wp_critical_max, .fwbs.dr_fw_wall, .tfcoil.dx_tf_wp_insulation |
| `.build.dr_tf_outboard` | 1b | `.stellarator.build` | second | 3 | .constraints.f_j_tf_wp_critical_max, .fwbs.dr_fw_wall, .tfcoil.dx_tf_wp_insulation |
| `.tfcoil.dr_tf_wp_with_insulation` | 1a | `.stellarator.coils.winding_pack_total_size_post` | second | 2 | .constraints.f_j_tf_wp_critical_max, .tfcoil.dx_tf_wp_insulation |
| `.tfcoil.dx_tf_wp_primary_toroidal` | 1a | `.stellarator.coils.winding_pack_total_size_post` | second | 2 | .constraints.f_j_tf_wp_critical_max, .tfcoil.dx_tf_wp_insulation |
| `.tfcoil.dx_tf_wp_secondary_toroidal` | 1a | `.stellarator.coils.winding_pack_total_size_post` | second | 2 | .constraints.f_j_tf_wp_critical_max, .tfcoil.dx_tf_wp_insulation |
| `.tfcoil.j_tf_wp` | 1a | `.stellarator.coils.winding_pack_total_size_post` | second | 2 | .constraints.f_j_tf_wp_critical_max, .tfcoil.dx_tf_wp_insulation |
| `.stellarator.wp_width_r_min` | 1a | `^problem.stellarator.coils.intersect` | second | 1 | .constraints.f_j_tf_wp_critical_max |

## Confirmed first-stage -- belief set 'sampled' (12)

| place | section | producing node |
|---|---|---|
| `.build.dr_cs` | 1d | `.initialisation.stellarator_solenoid_absent` |
| `.build.dr_cs_tf_gap` | 1d | `.initialisation.stellarator_solenoid_absent` |
| `.buildings.esbldgm3` | 1d | `.initialisation.energy_storage_building_volume` |
| `.physics.aspect` | 1a | `.stellarator.default_aspect_ratio` |
| `.stellarator.coilcurrent` | 1a | `.stellarator.coils.coil_current` |
| `.stellarator.f_st_i_total` | 1a | `.stellarator.coils.coil_current` |
| `.stellarator.r_coil_major` | 1a | `.stellarator.stellarator_scaling_factors` |
| `.stellarator.r_coil_minor` | 1a | `.stellarator.stellarator_scaling_factors` |
| `.tfcoil.dr_tf_plasma_case` | 1a | `.stellarator.coils.coil_casing` |
| `.tfcoil.dx_tf_side_case_min` | 1a | `.stellarator.coils.coil_casing` |
| `.tfcoil.n_tf_coils` | 1a | `.stellarator.stellarator_scaling_factors` |
| `.vacuum.d_duct` | 1b | `^problem.vacuum.duct_diameter_root_find` |

## Violations -- belief set 'build' (4)

| place | section | producing node | stage | # leaves | leaves |
|---|---|---|---|---|---|
| `.heat_transport.n_primary_heat_exchangers` | 1c | `.power.component_thermal_powers` | second | 18 | .costs.life_plant, .current_drive.eta_ecrh_injector_wall_plug, .divertor.tdiv, .fwbs.declblkt, .fwbs.declfw, .fwbs.f_p_blkt_multiplication, .heat_transport.f_p_fw_coolant_pump_total_heat, .impurity_radiation.f_nd_impurity_electron_array[13] +10 more |
| `.fwbs.life_fw_fpy` | 1b | `.stellarator.fwbs.fw_blanket_shield_geometry` | second | 13 | .costs.life_plant, .divertor.tdiv, .impurity_radiation.f_nd_impurity_electron_array[13], .physics.alphan, .physics.alphat, .physics.f_nd_alpha_thermal_electron, .physics.f_p_alpha_plasma_deposited, .physics.f_sync_reflect +5 more |
| `.vacuum.dia_vv_vacuum_ducts` | 1b | `.vacuum.vacuum_old` | second | 9 | .impurity_radiation.f_nd_impurity_electron_array[13], .physics.alphan, .physics.alphat, .physics.f_nd_alpha_thermal_electron, .physics.f_p_alpha_plasma_deposited, .physics.f_sync_reflect, .physics.f_temp_plasma_ion_electron, .physics.hfact +1 more |
| `.vacuum.n_vac_pumps_high` | 1b | `.vacuum.vacuum_old` | second | 9 | .impurity_radiation.f_nd_impurity_electron_array[13], .physics.alphan, .physics.alphat, .physics.f_nd_alpha_thermal_electron, .physics.f_p_alpha_plasma_deposited, .physics.f_sync_reflect, .physics.f_temp_plasma_ion_electron, .physics.hfact +1 more |

## Confirmed first-stage -- belief set 'build' (21)

| place | section | producing node |
|---|---|---|
| `.build.dr_cs` | 1d | `.initialisation.stellarator_solenoid_absent` |
| `.build.dr_cs_tf_gap` | 1d | `.initialisation.stellarator_solenoid_absent` |
| `.build.dr_fw_inboard` | 1b | `.stellarator.build` |
| `.build.dr_fw_outboard` | 1b | `.stellarator.build` |
| `.build.dr_shld_vv_gap_outboard` | 1b | `.stellarator.build` |
| `.build.dr_tf_outboard` | 1b | `.stellarator.build` |
| `.buildings.esbldgm3` | 1d | `.initialisation.energy_storage_building_volume` |
| `.physics.aspect` | 1a | `.stellarator.default_aspect_ratio` |
| `.stellarator.coilcurrent` | 1a | `.stellarator.coils.coil_current` |
| `.stellarator.f_st_i_total` | 1a | `.stellarator.coils.coil_current` |
| `.stellarator.r_coil_major` | 1a | `.stellarator.stellarator_scaling_factors` |
| `.stellarator.r_coil_minor` | 1a | `.stellarator.stellarator_scaling_factors` |
| `.stellarator.wp_width_r_min` | 1a | `^problem.stellarator.coils.intersect` |
| `.tfcoil.dr_tf_plasma_case` | 1a | `.stellarator.coils.coil_casing` |
| `.tfcoil.dr_tf_wp_with_insulation` | 1a | `.stellarator.coils.winding_pack_total_size_post` |
| `.tfcoil.dx_tf_side_case_min` | 1a | `.stellarator.coils.coil_casing` |
| `.tfcoil.dx_tf_wp_primary_toroidal` | 1a | `.stellarator.coils.winding_pack_total_size_post` |
| `.tfcoil.dx_tf_wp_secondary_toroidal` | 1a | `.stellarator.coils.winding_pack_total_size_post` |
| `.tfcoil.j_tf_wp` | 1a | `.stellarator.coils.winding_pack_total_size_post` |
| `.tfcoil.n_tf_coils` | 1a | `.stellarator.stellarator_scaling_factors` |
| `.vacuum.d_duct` | 1b | `^problem.vacuum.duct_diameter_root_find` |
