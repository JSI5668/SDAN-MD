
# SDAN-MD
-----------------------------------------------------------------------------------------------------------------------------
Supervised Dual Attention Network for Multi-Stage Motion Deblurring in Frontal-viewing Vehicle-camera Images. Any works that uses the provided pretrained network must acknowledge the authors by including the following reference.

    Seong In Jeong, Min Su Jeong,  Seon Jong Kang, Kyung Bong Ryu, and Kang Ryoung Park, “SDAN-MD: Supervised Dual Attention Network for Multi-Stage Motion Deblurring in Frontal-viewing Vehicle-camera Images,” King Saud University In submission 
    
<br>

-----------------------------------------------------------------------------------------------------------------------------

## Overall Architecture
<table>
  <tr>
    <td> <img src = "https://user-images.githubusercontent.com/79509777/205259623-5288699e-2209-4dd2-aa40-9122d30aed0d.png"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Overall Framework of SDAN-MD</b></p></td>
  </tr>
</table>

<table>
  <tr>
    <td> <img src = "https://user-images.githubusercontent.com/79509777/205260054-56dd572f-ceb7-4939-9e68-29124dbb3686.png" > </td>
  </tr>
  <tr>
    <td><p align="center"> <b>Supervised Dual Attention Module (SDAM)</b></p></td>
  </tr>
</table>


## Restoration Model

Download SDAN-MD model

CamVid      https://drive.google.com/file/d/1fRH-dcrelkSvr3Oi1P2qOq8vWSWfCbdN/view?usp=sharing

KITTI      https://drive.google.com/file/d/1PEfhOVzmFeauwWgzFx1a3O0T5P5b7YeS/view?usp=sharing


## Segmentation Model

CamVid      https://drive.google.com/file/d/14sTGeUKhQQMY6ejpiNpuAZz7uYHHo1rT/view?usp=sharing

KITTI       https://drive.google.com/file/d/1thXhIjQXJBoF7JRlWAgSeAycoqLSnpyF/view?usp=sharing


## Download CamVid database (11 classes)

https://drive.google.com/file/d/1-ugLiHit40BWfJwovZDNoyaL2YuKjXDy/view?usp=sharing

## Download KITTI database (11 classes)

https://drive.google.com/file/d/106iM9GJhbqlMLKtpzcp260a67E-tncJo/view?usp=sharing

-----------------------------------------------------------------------------------------------------------------------------

## Prerequisites

- python 3.8.13 
- pytorch 1.12.1
- Windows 10

-----------------------------------------------------------------------------------------------------------------------------

## Reference


- [1]  Brostow, G. J.; Fauqueur, J.; Cipolla, R., Semantic object classes in video: A high-definition ground truth database. Pattern Recognit. Lett. 2009, 30, (2); pp. 88-97.

- [2]  Krešo, I.; Čaušević, D.; Krapac, J.; Šegvić, S. Convolutional scale invariance for semantic segmentation. In proceedings of the German Conference on Pattern Recognition (GCPR), Bonn, Germany 28 September-1 October 2016; pp. 64-75
