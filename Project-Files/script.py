from DataHolder import DataHolder, Processor
from ModelAgent import UMAP, VAE_model

### Client loader
Glitne = DataHolder("Glitne", [1300, 1502, 2], [1500, 2002, 2])
Glitne.add_segy('near', '../data/3d_nearstack.sgy');
Glitne.add_segy('far', '../data/3d_farstack.sgy');
Glitne.add_horizon('top_heimdal', '../data/Top_Heimdal_subset.txt')
Glitne.add_well('well_1', 36, 276//2)

### Client data factory - would run diffetent instances of Processing with different operations
# instance of processing creates options for many outputs
Data = Processor(Glitne.near, Glitne.far, Glitne.twt)
processing_a = Data([Glitne.horizons['top_heimdal'], 12,52], [10, 20])
processing_b = Data([Glitne.horizons['top_heimdal'], 10,10], [10, 20])

### Client model creator/run - run many different instances of the VAE with different parameters
# an instance of a model is one model of dim reduction
UMAP_a = UMAP(processing_a)
UMAP_a1 = UMAP_a.reduce(n_neighbors=10)
UMAP_a2 = UMAP_a.reduce(n_neighbors=100)

# Run instance of VAE model
VAE_a = VAE_model(processing_a)
VAE_a.reduce()
