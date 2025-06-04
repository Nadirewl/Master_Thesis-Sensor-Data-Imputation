from util2s.moodel import cph_ceon_tags
for var_name, var in vars(cph_ceon_tags).items():
    print(var_name, getattr(var, 'base_tag', None))