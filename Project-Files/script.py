from main import MasterClass, loss_function, train, test, forward_all


object = MasterClass('TEST_OBJECT')

# Loading Data
object.load_near('../data/3d_nearstack.sgy')
object.load_far('../data/3d_farstack.sgy')
object.load_horizon('../data/Top_Heimdal_subset.txt')

object.plot_horizon()

# add well location
object.add_well('well_1', 36, 276//2)

# print well dictionary and plot
# print("Dictionary of wells", object.wells)
# object.plot_horizon_well('well_1')

# flatten
object.flatten_traces()

# normalise and reshape to 2d arrays
object.normalise_from_well()
#
# # linear regression to find FF:
object.get_FF()

# Stack the near and far on top of each-other
object.stack_traces()

# # Umap dimensionality reduction:
# object.umap()
# object.plot_umap()

print("UMAP Script run successfully")

print("Start VAE testing")

object.create_dataloader()
# object.train_vae()
object.run_vae()
object.vae_umap()

print("VAE finished no bugs")

