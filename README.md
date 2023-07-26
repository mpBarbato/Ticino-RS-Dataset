# Ticino-RS-Dataset

Ticino dataset is a novel multi-modal remote sensing dataset specifically tailored for semantic segmentation tasks. It covers an area of about 1332 $km^2$.
<br />
<br />
It incorporates five distinct modalities:
- **RGB**
  - spatial resolution: 1.86m/px (vertical) - 2.64m/px (horizontal)
  - 3 bands 
- **Digital Terrain Model** (elevation of the ground)
  - spatial resolution: 5m/px
  - 1 band
  - from 51.86 to 124.75 meters
- **Panchromatic**
  - spatial resolution: 5m/px
  - 1 band [400-700 nm]
- **Hyperspectral**
  - **VNIR**
    - spatial resolution: 30m/px &#8594; 5m/px (pansharpened)
    - 63 bands [400-1010 nm] &#8594; 60 bands (cleaned)
  - **SWIR**
    - spatial resolution: 30m/px &#8594; 5m/px (pansharpened)
    - 171 bands [920-2500 nm] &#8594; 122 bands (cleaned)
   
         
It includes two pixel-level labelings:
- **Land Cover**:
  - 8 classes:
    - 0 Background
    - 1 Building
    - 2 Road
    - 3 Residential
    - 4 Industrial
    - 5 Forest
    - 6 Farmland
    - 7 Water
- **Soil Agricultural Use**:
  - 10 classes:
    - 0 Background
    - 1 Other agricultural crops
    - 2 Forage crops
    - 3 Corn
    - 4 Industrial plants
    - 5 Rice
    - 6 Seeds
    - 7 Man-made areas
    - 8 Water bodies
    - 9 Natural vegetation

## How to train a model?

## How to test a model?

## Citation
