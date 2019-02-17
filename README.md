# A Hierarchical Deep Learning Approach for In-loop Filtering in Video Coding
------
The paper proposes a deep learning based systematic approach that includes an effective Convolutional Neural Network (CNN) structure, a hierarchical training strategy, and a video codec oriented switchable mechanism. In brief, the contributions of this work are as follows: 

- A novel network,named as ***Squeezeand-Excitation Filtering CNN (SEFCNN)***, is designed, which is comprised of two subnets: *Feature EXtracting (FEX) net and Feature ENhancing (FEN) net*. The FEX is a stack of convolutional layers characterizing the spatial and channel-wise correlation, while the FEN is a squeeze-andexcitation net that fully explores the relationship between channels. 
- ***A hierarchical model training strategy*** is developed. During the encoding, (a) different Quantization Parameters (QPs) cause different levels of artifacts; (b) different frame types employ different coding tools and thus exhibit different artifact proprieties. In contrast to prior researches that design a single powerful network for all kinds of artifacts, we propose to **hierarchically deploy two subnets for different coding scenarios**. 
-  When incorporating CNN model into video encoder, ***we conduct an adaptive mechanism that switches between the CNN-based and the traditional methods to selectively enhance some frames or some regions of a frame.*** Compared to previous work that applies one model to every single frame, our approach takes advantage of coding reference structure and obtains the superiority in both encoder computational complexity and overall coding efﬁciency. 
## State-of-the-art solutions 
   ![State-of-the-art solutions ](https://github.com/IVC-Projects/cnn_In-loop_filter/blob/master/figures/fig1.png)
## Network Structure
   ![Network Structure](https://github.com/IVC-Projects/cnn_In-loop_filter/blob/master/figures/network%20structrue.png)
## Hierarchical CNN Models for Different Frame Types 
   ![Hierarchical CNN Models for Different Frame Types](https://github.com/IVC-Projects/cnn_In-loop_filter/blob/master/figures/table6.png)
   ![fig 6](https://github.com/IVC-Projects/cnn_In-loop_filter/blob/master/figures/fig6.png)
##  visual comparisons 
   ![visual comparisons](https://github.com/IVC-Projects/cnn_In-loop_filter/blob/master/figures/fig7.png)
##  EXPERIMENTAL RESULTS 
  ![CODING PERFORMANCE COMPARED WITH VRCNN (BD-RATE)
](https://github.com/IVC-Projects/cnn_In-loop_filter/blob/master/figures/table7.png)
 ![CODING PERFORMANCE OF THE PROPOSED SEFCNN COMPARED TO VRCNN AND RHCNN (QP = 37)
](https://github.com/IVC-Projects/cnn_In-loop_filter/blob/master/figures/table8.png)
 ![CODING PERFORMANCE COMPARED WITH WORK [39] (BD-RATE COMPARED TO HM12.0)
](https://github.com/IVC-Projects/cnn_In-loop_filter/blob/master/figures/table9.png)
 ![ PSNR histogram statistics of different network structures on common test sequences.
](https://github.com/IVC-Projects/cnn_In-loop_filter/blob/master/figures/fig8.png)
 ![CODING PERFORMANCE COMPARED WITH WORK [40] (BD-RATE COMPARED TO HM7.0)
](https://github.com/IVC-Projects/cnn_In-loop_filter/blob/master/figures/table10.png)
 ![Visualization of the parameter quantity versus PSNR gain in various network deployments.
](https://github.com/IVC-Projects/cnn_In-loop_filter/blob/master/figures/fig9.png)
 ![AVERAGE ENCODING TIME COMPARED TO HM16.9 (SECOND/FRAME)
](https://github.com/IVC-Projects/cnn_In-loop_filter/blob/master/figures/table11.png)
