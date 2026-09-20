"""One explicit model choice shared by training and checkpoint inference."""

def make_training_model(input_ph, is_training_ph, num_out, arch_code, component_variant=None):
    if component_variant is None:
        from efnas.network.fixed_arch_models import FixedArchModelV3
        return FixedArchModelV3(input_ph=input_ph, is_training_ph=is_training_ph,
                                arch_code=arch_code, num_out=num_out,
                                init_neurons=32, expansion_factor=2.0)
    from efnas.network.ablation_edgeflownet import ABlationEdgeFlowNetV1, build_ablation_variants
    variant = build_ablation_variants([component_variant])[0]
    return ABlationEdgeFlowNetV1(input_ph=input_ph, is_training_ph=is_training_ph,
                                variant_config=variant, num_out=num_out,
                                init_neurons=32, expansion_factor=2.0)
