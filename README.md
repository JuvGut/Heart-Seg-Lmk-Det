# Heart-Valve-Segmentor

## To do
- [x] Look at data
- [x] Create a GitLab-repo
- [x] make DataLoader faster
- [x] (Work with MONAI)
- [x] Find work-around of using differently structured labels (background is last, instead of first label)
- [ ] Train **[nnU-Net](https://github.com/MIC-DKFZ/nnUNet)** to segment the heart valves ~~-> DynUNet (monai implementation of nnU-Net) "Project Monai folder on GitLab. Runs but takes 22 min for validation"~~
- [ ] **SEE BELOW: What to train**
- [ ] Create full size dataset (with all the data)

- [ ] Train a 3D-landmark detector to detect the landmarks
- [ ] Deploy the trained models to Specto
- [ ] Write and complete the README file.

### Training on small "test"-Dataset
- [x] 2D U-Net: Fold 0, 1, 2, 3, 4
- [x] 3D full resolution U-Net: 0
- [ ] 3D full resolution U-Net: 1, 2, 3, 4
- [ ] 3D full resolution U-Net: 0, 1, 2, 3, 4
- [ ] 3D U-Net cascade: 0, 1, 2, 3, 4
- [ ] 3D full resolution U-Net: 0, 1, 2, 3, 4

### Goal: **FIND BEST CONFIGURATION**
- [ ] [TO DO (How to use nnU-Net)](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md)

## README:
### Description
This project aims to segment the heart into five segments and also to detect landmarks in CT scans of hearts, anonymized and provided by the [University Hospital of Basel](https://www.unispital-basel.ch/).

The segments are:
- LVOT
- RCC
- LCC
- NCC
- BG

The landmarks are:
- Commissure LCC-RCC
- Commissure LCC-NCC
- Commissure NCC-RCC
- Nadir RC
- Nadir NC
- Basal NCC-RCC

### Visuals
![alt text](image.png)
### Installation
### Usage
### Roadmap
### Authors and acknowledgment
Juval Gutknecht

### License
[Center for medical Image Analysis and Navigation, Department of Biomedical Engineering, University of Basel](https://dbe.unibas.ch/en/cian/)