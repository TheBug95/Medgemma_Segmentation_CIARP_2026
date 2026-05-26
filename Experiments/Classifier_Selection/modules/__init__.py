# =============================================================================
# modules/__init__.py
# =============================================================================
# Módulo de datos para el experimento de selección de clasificador.
#
# PROVEE:
#   - DataModule: Clase principal para cargar datos REFUGE
#
# USO:
#   from modules.data_module import DataModule
#
# INSTANCIAR:
#   data_module = DataModule(config)
#   train_loader = data_module.get_train_loader()
#   val_loader = data_module.get_val_loader()
# =============================================================================