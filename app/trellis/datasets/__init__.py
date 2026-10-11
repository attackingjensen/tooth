import importlib

__attributes = {
    'SparseStructure': 'sparse_structure',
    
    'SparseFeat2Render': 'sparse_feat2render',
    'SLat2Render':'structured_latent2render',
    'Slat2RenderGeo':'structured_latent2render',
    
    'SparseStructureLatent': 'sparse_structure_latent',
    'TextConditionedSparseStructureLatent': 'sparse_structure_latent',
    'ImageConditionedSparseStructureLatent': 'sparse_structure_latent',
    
    'SLat': 'structured_latent',
    'TextConditionedSLat': 'structured_latent',
    'ImageConditionedSLat': 'structured_latent',

    'EditingTextSparseStructureLatent': 'editing_text_latent',

    'EditingSLatText': 'editing_slat_text',

    'Editing3DEditFormerImageSparseStructureLatent': 'editing_3deditformer_image',
    'Editing3DEditFormerSLatImage': 'editing_3deditformer_image',
    'CrownJawSS': 'crown_ss',
    'CrownJawInpaintSS': 'crown_ss',
    'CrownJawPointSS': 'crown_point',
    'CrownTemplateJawSS': 'crown_template_jaw_ss',
    'CrownTextJawSS': 'crown_text_jaw_ss',
    'CrownJawSLat': 'crown_slat',
    'CrownJawSLatCross': 'crown_slat',
    'CrownJawInpaintSLat': 'crown_slat',
    'CrownTextJawSLat': 'crown_text_jaw_slat',

    'H3DOriImageSparseStructureLatent': 'editing_h3d_image',
    'H3DOriSLatImage': 'editing_h3d_image',
    'H3DOriSLatImageTokenConcat': 'editing_h3d_image',
    'MixedH3D3DEditVerseOriImageSparseStructureLatent': 'editing_h3d_image',
    'MixedH3D3DEditVerseOriSLatImage': 'editing_h3d_image',
    'MixedH3D3DEditVerseOriSLatImageTokenConcat': 'editing_h3d_image',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':
    from .sparse_structure import SparseStructure
    
    from .sparse_feat2render import SparseFeat2Render
    from .structured_latent2render import (
        SLat2Render,
        Slat2RenderGeo,
    )
    
    from .sparse_structure_latent import (
        SparseStructureLatent,
        TextConditionedSparseStructureLatent,
        ImageConditionedSparseStructureLatent,
    )
    
    from .structured_latent import (
        SLat,
        TextConditionedSLat,
        ImageConditionedSLat,
    )

    from .editing_text_latent import EditingTextSparseStructureLatent

    from .editing_slat_text import EditingSLatText

    from .editing_3deditformer_image import (
        Editing3DEditFormerImageSparseStructureLatent,
        Editing3DEditFormerSLatImage,
    )

    from .editing_h3d_image import (
        H3DOriImageSparseStructureLatent,
        H3DOriSLatImage,
        MixedH3D3DEditVerseOriImageSparseStructureLatent,
    )
    
