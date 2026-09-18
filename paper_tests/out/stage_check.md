# Stage check: non-anticipativity of claimed build outputs

## Summary

| belief set | leaves | nodes 1st | nodes 2nd | owned 1st | owned 2nd | 1st-stage frac | violations |
|---|---|---|---|---|---|---|---|
| all | 147 | 27 | 128 | 92 | 449 | 0.17 | 13 |
| sampled | 26 | 36 | 120 | 103 | 441 | 0.23 | 13 |
| build | 22 | 60 | 96 | 191 | 353 | 0.38 | 4 |

## Violations -- belief set 'all' (13)

| place | kind | source | producing node | # leaves | leaves |
|---|---|---|---|---|---|
| `.heat_transport.n_primary_heat_exchangers` | sizing choice | output_kinds.md §1c | `.power.component_thermal_powers` | 53 | .constraints.f_j_tf_wp_critical_max, .costs.abktflnc, .current_drive.eta_ecrh_injector_wall_plug, .current_drive.p_hcd_primary_extra_heat_mw, .divertor.tdiv, .divertor.xpertin, .fwbs.declblkt, .fwbs.declfw +45 more |
| `.fwbs.life_fw_fpy` | §3.6 | decision_kinds.md §3.6 | `.stellarator.fwbs.fw_blanket_shield_geometry` | 42 | .constraints.f_j_tf_wp_critical_max, .costs.abktflnc, .divertor.tdiv, .divertor.xpertin, .fwbs.fvolsi, .fwbs.fvolso, .impurity_radiation.f_nd_impurity_electron_array[10], .impurity_radiation.f_nd_impurity_electron_array[11] +34 more |
| `.vacuum.dia_vv_vacuum_ducts` | sizing choice | output_kinds.md §1b | `.vacuum.vacuum_old` | 36 | .constraints.f_j_tf_wp_critical_max, .impurity_radiation.f_nd_impurity_electron_array[10], .impurity_radiation.f_nd_impurity_electron_array[11], .impurity_radiation.f_nd_impurity_electron_array[12], .impurity_radiation.f_nd_impurity_electron_array[13], .impurity_radiation.f_nd_impurity_electron_array[2], .impurity_radiation.f_nd_impurity_electron_array[3], .impurity_radiation.f_nd_impurity_electron_array[4] +28 more |
| `.vacuum.n_vac_pumps_high` | sizing choice | output_kinds.md §1b | `.vacuum.vacuum_old` | 36 | .constraints.f_j_tf_wp_critical_max, .impurity_radiation.f_nd_impurity_electron_array[10], .impurity_radiation.f_nd_impurity_electron_array[11], .impurity_radiation.f_nd_impurity_electron_array[12], .impurity_radiation.f_nd_impurity_electron_array[13], .impurity_radiation.f_nd_impurity_electron_array[2], .impurity_radiation.f_nd_impurity_electron_array[3], .impurity_radiation.f_nd_impurity_electron_array[4] +28 more |
| `.build.dr_fw_inboard` | §3.6 | decision_kinds.md §3.6 | `.stellarator.build` | 2 | .constraints.f_j_tf_wp_critical_max, .tfcoil.tftmp |
| `.build.dr_shld_vv_gap_outboard` | sizing choice | output_kinds.md §1b | `.stellarator.build` | 2 | .constraints.f_j_tf_wp_critical_max, .tfcoil.tftmp |
| `.build.dr_tf_inboard` | §3.6 | decision_kinds.md §3.6 | `.stellarator.coils.coil_radial_thickness` | 2 | .constraints.f_j_tf_wp_critical_max, .tfcoil.tftmp |
| `.build.dr_tf_outboard` | sizing choice | output_kinds.md §1b | `.stellarator.build` | 2 | .constraints.f_j_tf_wp_critical_max, .tfcoil.tftmp |
| `.stellarator.wp_width_r_min` | sizing choice | output_kinds.md §1a | `^problem.stellarator.coils.intersect` | 2 | .constraints.f_j_tf_wp_critical_max, .tfcoil.tftmp |
| `.tfcoil.dr_tf_wp_with_insulation` | §3.6 | decision_kinds.md §3.6 | `.stellarator.coils.winding_pack_total_size_post` | 2 | .constraints.f_j_tf_wp_critical_max, .tfcoil.tftmp |
| `.tfcoil.dx_tf_wp_primary_toroidal` | definition | output_kinds.md §1a | `.stellarator.coils.winding_pack_total_size_post` | 2 | .constraints.f_j_tf_wp_critical_max, .tfcoil.tftmp |
| `.tfcoil.e_tf_magnetic_stored_total_gj` | §3.6 | decision_kinds.md §3.6 | `.stellarator.coils.stored_magnetic_energy` | 2 | .constraints.f_j_tf_wp_critical_max, .tfcoil.tftmp |
| `.tfcoil.j_tf_wp` | sizing choice | output_kinds.md §1a | `.stellarator.coils.winding_pack_total_size_post` | 2 | .constraints.f_j_tf_wp_critical_max, .tfcoil.tftmp |

## Confirmed first-stage -- belief set 'all' (13)

| place | kind | source | producing node |
|---|---|---|---|
| `.build.dr_cs` | §3.6 | decision_kinds.md §3.6 | `.initialisation.stellarator_solenoid_absent` |
| `.build.z_tf_inside_half` | §3.6 | decision_kinds.md §3.6 | `.stellarator.coils.z_tf_inside_half` |
| `.buildings.esbldgm3` | definition | output_kinds.md §1d | `.initialisation.energy_storage_building_volume` |
| `.physics.aspect` | §3.6 | decision_kinds.md §3.6 | `.stellarator.default_aspect_ratio` |
| `.physics.eps` | §3.6 | decision_kinds.md §3.6 | `.stellarator.stellarator_scaling_factors` |
| `.physics.rminor` | §3.6 | decision_kinds.md §3.6 | `.stellarator.stellarator_scaling_factors` |
| `.stellarator.coilcurrent` | §3.6 | decision_kinds.md §3.6 | `.stellarator.coils.coil_current` |
| `.stellarator.r_coil_major` | §3.6 | decision_kinds.md §3.6 | `.stellarator.stellarator_scaling_factors` |
| `.tfcoil.dr_tf_plasma_case` | §3.6 | decision_kinds.md §3.6 | `.stellarator.coils.coil_casing` |
| `.tfcoil.dx_tf_side_case_min` | sizing choice | output_kinds.md §1a | `.stellarator.coils.coil_casing` |
| `.tfcoil.n_tf_coils` | §3.6 | decision_kinds.md §3.6 | `.stellarator.stellarator_scaling_factors` |
| `.times.t_plant_pulse_burn` | §3.6 | decision_kinds.md §3.6 | `.initialisation.stellarator_pulse_times` |
| `.vacuum.d_duct` | sizing choice | output_kinds.md §1b | `^problem.vacuum.duct_diameter_root_find` |

## Violations -- belief set 'sampled' (13)

| place | kind | source | producing node | # leaves | leaves |
|---|---|---|---|---|---|
| `.heat_transport.n_primary_heat_exchangers` | sizing choice | output_kinds.md §1c | `.power.component_thermal_powers` | 22 | .constraints.f_j_tf_wp_critical_max, .costs.life_plant, .current_drive.eta_ecrh_injector_wall_plug, .divertor.tdiv, .fwbs.declblkt, .fwbs.declfw, .fwbs.dr_fw_wall, .fwbs.f_p_blkt_multiplication +14 more |
| `.fwbs.life_fw_fpy` | §3.6 | decision_kinds.md §3.6 | `.stellarator.fwbs.fw_blanket_shield_geometry` | 17 | .constraints.f_j_tf_wp_critical_max, .costs.life_plant, .divertor.tdiv, .fwbs.dr_fw_wall, .fwbs.fhole, .impurity_radiation.f_nd_impurity_electron_array[13], .physics.alphan, .physics.alphat +9 more |
| `.vacuum.dia_vv_vacuum_ducts` | sizing choice | output_kinds.md §1b | `.vacuum.vacuum_old` | 12 | .constraints.f_j_tf_wp_critical_max, .fwbs.dr_fw_wall, .impurity_radiation.f_nd_impurity_electron_array[13], .physics.alphan, .physics.alphat, .physics.f_nd_alpha_thermal_electron, .physics.f_p_alpha_plasma_deposited, .physics.f_sync_reflect +4 more |
| `.vacuum.n_vac_pumps_high` | sizing choice | output_kinds.md §1b | `.vacuum.vacuum_old` | 12 | .constraints.f_j_tf_wp_critical_max, .fwbs.dr_fw_wall, .impurity_radiation.f_nd_impurity_electron_array[13], .physics.alphan, .physics.alphat, .physics.f_nd_alpha_thermal_electron, .physics.f_p_alpha_plasma_deposited, .physics.f_sync_reflect +4 more |
| `.build.dr_fw_inboard` | §3.6 | decision_kinds.md §3.6 | `.stellarator.build` | 3 | .constraints.f_j_tf_wp_critical_max, .fwbs.dr_fw_wall, .tfcoil.dx_tf_wp_insulation |
| `.build.dr_shld_vv_gap_outboard` | sizing choice | output_kinds.md §1b | `.stellarator.build` | 3 | .constraints.f_j_tf_wp_critical_max, .fwbs.dr_fw_wall, .tfcoil.dx_tf_wp_insulation |
| `.build.dr_tf_outboard` | sizing choice | output_kinds.md §1b | `.stellarator.build` | 3 | .constraints.f_j_tf_wp_critical_max, .fwbs.dr_fw_wall, .tfcoil.dx_tf_wp_insulation |
| `.build.dr_tf_inboard` | §3.6 | decision_kinds.md §3.6 | `.stellarator.coils.coil_radial_thickness` | 2 | .constraints.f_j_tf_wp_critical_max, .tfcoil.dx_tf_wp_insulation |
| `.tfcoil.dr_tf_wp_with_insulation` | §3.6 | decision_kinds.md §3.6 | `.stellarator.coils.winding_pack_total_size_post` | 2 | .constraints.f_j_tf_wp_critical_max, .tfcoil.dx_tf_wp_insulation |
| `.tfcoil.dx_tf_wp_primary_toroidal` | definition | output_kinds.md §1a | `.stellarator.coils.winding_pack_total_size_post` | 2 | .constraints.f_j_tf_wp_critical_max, .tfcoil.dx_tf_wp_insulation |
| `.tfcoil.e_tf_magnetic_stored_total_gj` | §3.6 | decision_kinds.md §3.6 | `.stellarator.coils.stored_magnetic_energy` | 2 | .constraints.f_j_tf_wp_critical_max, .tfcoil.dx_tf_wp_insulation |
| `.tfcoil.j_tf_wp` | sizing choice | output_kinds.md §1a | `.stellarator.coils.winding_pack_total_size_post` | 2 | .constraints.f_j_tf_wp_critical_max, .tfcoil.dx_tf_wp_insulation |
| `.stellarator.wp_width_r_min` | sizing choice | output_kinds.md §1a | `^problem.stellarator.coils.intersect` | 1 | .constraints.f_j_tf_wp_critical_max |

## Confirmed first-stage -- belief set 'sampled' (13)

| place | kind | source | producing node |
|---|---|---|---|
| `.build.dr_cs` | §3.6 | decision_kinds.md §3.6 | `.initialisation.stellarator_solenoid_absent` |
| `.build.z_tf_inside_half` | §3.6 | decision_kinds.md §3.6 | `.stellarator.coils.z_tf_inside_half` |
| `.buildings.esbldgm3` | definition | output_kinds.md §1d | `.initialisation.energy_storage_building_volume` |
| `.physics.aspect` | §3.6 | decision_kinds.md §3.6 | `.stellarator.default_aspect_ratio` |
| `.physics.eps` | §3.6 | decision_kinds.md §3.6 | `.stellarator.stellarator_scaling_factors` |
| `.physics.rminor` | §3.6 | decision_kinds.md §3.6 | `.stellarator.stellarator_scaling_factors` |
| `.stellarator.coilcurrent` | §3.6 | decision_kinds.md §3.6 | `.stellarator.coils.coil_current` |
| `.stellarator.r_coil_major` | §3.6 | decision_kinds.md §3.6 | `.stellarator.stellarator_scaling_factors` |
| `.tfcoil.dr_tf_plasma_case` | §3.6 | decision_kinds.md §3.6 | `.stellarator.coils.coil_casing` |
| `.tfcoil.dx_tf_side_case_min` | sizing choice | output_kinds.md §1a | `.stellarator.coils.coil_casing` |
| `.tfcoil.n_tf_coils` | §3.6 | decision_kinds.md §3.6 | `.stellarator.stellarator_scaling_factors` |
| `.times.t_plant_pulse_burn` | §3.6 | decision_kinds.md §3.6 | `.initialisation.stellarator_pulse_times` |
| `.vacuum.d_duct` | sizing choice | output_kinds.md §1b | `^problem.vacuum.duct_diameter_root_find` |

## Violations -- belief set 'build' (4)

| place | kind | source | producing node | # leaves | leaves |
|---|---|---|---|---|---|
| `.heat_transport.n_primary_heat_exchangers` | sizing choice | output_kinds.md §1c | `.power.component_thermal_powers` | 18 | .costs.life_plant, .current_drive.eta_ecrh_injector_wall_plug, .divertor.tdiv, .fwbs.declblkt, .fwbs.declfw, .fwbs.f_p_blkt_multiplication, .heat_transport.f_p_fw_coolant_pump_total_heat, .impurity_radiation.f_nd_impurity_electron_array[13] +10 more |
| `.fwbs.life_fw_fpy` | §3.6 | decision_kinds.md §3.6 | `.stellarator.fwbs.fw_blanket_shield_geometry` | 13 | .costs.life_plant, .divertor.tdiv, .impurity_radiation.f_nd_impurity_electron_array[13], .physics.alphan, .physics.alphat, .physics.f_nd_alpha_thermal_electron, .physics.f_p_alpha_plasma_deposited, .physics.f_sync_reflect +5 more |
| `.vacuum.dia_vv_vacuum_ducts` | sizing choice | output_kinds.md §1b | `.vacuum.vacuum_old` | 9 | .impurity_radiation.f_nd_impurity_electron_array[13], .physics.alphan, .physics.alphat, .physics.f_nd_alpha_thermal_electron, .physics.f_p_alpha_plasma_deposited, .physics.f_sync_reflect, .physics.f_temp_plasma_ion_electron, .physics.hfact +1 more |
| `.vacuum.n_vac_pumps_high` | sizing choice | output_kinds.md §1b | `.vacuum.vacuum_old` | 9 | .impurity_radiation.f_nd_impurity_electron_array[13], .physics.alphan, .physics.alphat, .physics.f_nd_alpha_thermal_electron, .physics.f_p_alpha_plasma_deposited, .physics.f_sync_reflect, .physics.f_temp_plasma_ion_electron, .physics.hfact +1 more |

## Confirmed first-stage -- belief set 'build' (22)

| place | kind | source | producing node |
|---|---|---|---|
| `.build.dr_cs` | §3.6 | decision_kinds.md §3.6 | `.initialisation.stellarator_solenoid_absent` |
| `.build.dr_fw_inboard` | §3.6 | decision_kinds.md §3.6 | `.stellarator.build` |
| `.build.dr_shld_vv_gap_outboard` | sizing choice | output_kinds.md §1b | `.stellarator.build` |
| `.build.dr_tf_inboard` | §3.6 | decision_kinds.md §3.6 | `.stellarator.coils.coil_radial_thickness` |
| `.build.dr_tf_outboard` | sizing choice | output_kinds.md §1b | `.stellarator.build` |
| `.build.z_tf_inside_half` | §3.6 | decision_kinds.md §3.6 | `.stellarator.coils.z_tf_inside_half` |
| `.buildings.esbldgm3` | definition | output_kinds.md §1d | `.initialisation.energy_storage_building_volume` |
| `.physics.aspect` | §3.6 | decision_kinds.md §3.6 | `.stellarator.default_aspect_ratio` |
| `.physics.eps` | §3.6 | decision_kinds.md §3.6 | `.stellarator.stellarator_scaling_factors` |
| `.physics.rminor` | §3.6 | decision_kinds.md §3.6 | `.stellarator.stellarator_scaling_factors` |
| `.stellarator.coilcurrent` | §3.6 | decision_kinds.md §3.6 | `.stellarator.coils.coil_current` |
| `.stellarator.r_coil_major` | §3.6 | decision_kinds.md §3.6 | `.stellarator.stellarator_scaling_factors` |
| `.stellarator.wp_width_r_min` | sizing choice | output_kinds.md §1a | `^problem.stellarator.coils.intersect` |
| `.tfcoil.dr_tf_plasma_case` | §3.6 | decision_kinds.md §3.6 | `.stellarator.coils.coil_casing` |
| `.tfcoil.dr_tf_wp_with_insulation` | §3.6 | decision_kinds.md §3.6 | `.stellarator.coils.winding_pack_total_size_post` |
| `.tfcoil.dx_tf_side_case_min` | sizing choice | output_kinds.md §1a | `.stellarator.coils.coil_casing` |
| `.tfcoil.dx_tf_wp_primary_toroidal` | definition | output_kinds.md §1a | `.stellarator.coils.winding_pack_total_size_post` |
| `.tfcoil.e_tf_magnetic_stored_total_gj` | §3.6 | decision_kinds.md §3.6 | `.stellarator.coils.stored_magnetic_energy` |
| `.tfcoil.j_tf_wp` | sizing choice | output_kinds.md §1a | `.stellarator.coils.winding_pack_total_size_post` |
| `.tfcoil.n_tf_coils` | §3.6 | decision_kinds.md §3.6 | `.stellarator.stellarator_scaling_factors` |
| `.times.t_plant_pulse_burn` | §3.6 | decision_kinds.md §3.6 | `.initialisation.stellarator_pulse_times` |
| `.vacuum.d_duct` | sizing choice | output_kinds.md §1b | `^problem.vacuum.duct_diameter_root_find` |
