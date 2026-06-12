import pymeshlab
print(pymeshlab.number_plugins())
ms = pymeshlab.MeshSet()
ms.load_new_mesh("poc/hrtf/outputs/FABIAN_6k_HATO0_truncated.stl")

# List all available filters
filters = [f for f in dir(ms) if not f.startswith('_')]
print([f for f in filters if 'quality' in f.lower()])
print([f for f in filters if 'decim' in f.lower()])
print([f for f in filters if 'scalar' in f.lower()])

# Check mesh methods
m = ms.current_mesh()
methods = [x for x in dir(m) if not x.startswith('_')]
print([x for x in methods if 'quality' in x.lower() or 'scalar' in x.lower()])