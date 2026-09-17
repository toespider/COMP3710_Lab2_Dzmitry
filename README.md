Contains all code for Demo 2 of COMP3710
Contains all demo results

<img width="1198" height="796" alt="Screenshot 2026-09-18 at 8 27 45 am" src="https://github.com/user-attachments/assets/9e83f6e6-1159-4f78-ba27-7ec0e17f65b3" />


<img width="1198" height="796" alt="image" src="https://github.com/user-attachments/assets/e4dc3c85-ad19-4f09-ad15-2f269da0adae" />

--- DFT/FFT Performance Comparison ---
1. Built-in FFT Execution Time:      0.001092 seconds
2. Vectorized GPU DFT Execution Time:0.001380 seconds
3. Naive Loop DFT Execution Time:    13.641975 seconds

GPU DFT matches Built-in FFT: False

Direct DFT is O(N^2), but FFT is O(NlogN)
Native loop is two nested for loops, so O(N^2)

Although the vectorized DFT uses the GPU to perform many calculations in parallel, it still requires \(O(N^2)\) operations because it constructs and multiplies by the complete DFT transformation matrix.


<img width="720" height="719" alt="Figure_1" src="https://github.com/user-attachments/assets/87d82562-af3a-49a7-81cb-ab630313a68b" />

<img width="635" height="475" alt="Screenshot 2026-09-18 at 8 50 25 am" src="https://github.com/user-attachments/assets/2fea11a1-8575-41b8-af35-533230c16460" />


 Total dataset size :
 n_samples : 1288
 n_features : 1850
 n_classes : 7
(966, 150)
(322, 150)
(966,)
 Total Testing  322
 Predictions  [3 3 6 3 3 3 3 1 3 3 3 3 3 3 3 3 3 3 3 4 1 1 3 3 3 1 3 3 3 3 3 3 3 3 3 3 3
 3 3 1 3 1 3 1 1 3 3 3 4 3 3 3 3 3 1 2 1 3 5 3 6 1 3 6 3 5 3 4 1 3 6 4 3 3
 3 2 3 3 3 3 6 3 3 3 3 3 3 3 3 1 3 3 3 3 1 3 1 1 2 6 3 3 3 3 3 3 3 3 3 1 3
 1 6 3 3 3 1 4 1 3 1 3 3 1 3 4 5 3 1 3 6 6 6 3 3 4 3 3 1 3 3 3 3 1 3 3 1 3
 6 1 4 3 1 3 3 3 3 3 3 3 4 5 5 1 3 3 5 1 3 3 3 3 3 1 5 3 3 3 3 5 3 3 3 1 3
 3 3 3 3 2 4 3 2 3 6 3 3 3 3 3 3 3 3 3 5 1 4 2 4 3 1 5 4 3 3 3 3 2 3 3 3 6
 3 1 1 3 3 3 3 3 3 3 3 3 3 1 1 6 3 3 3 4 1 3 3 3 3 3 3 4 4 4 3 4 3 4 3 1 3
 3 3 3 3 1 3 6 6 1 6 1 1 3 3 3 6 3 3 3 3 3 1 1 3 3 3 3 3 3 3 4 3 3 5 3 3 1
 3 5 3 3 3 6 3 3 1 3 3 3 1 3 3 3 1 3 3 3 3 1 3 3 4 3]
 Which Correct : [ True  True  True  True  True  True False  True  True  True  True  True
  True False  True  True  True  True  True  True  True False  True False
 False  True False  True  True  True False  True  True  True  True  True
  True  True  True  True  True  True  True  True  True False False  True
 False  True  True  True False  True False  True  True  True  True  True
 False  True False False  True  True False  True  True  True False False
  True  True  True  True False False False False False False False  True
  True  True  True  True  True False  True False  True False  True False
  True  True False  True  True False  True False  True  True  True  True
  True  True False  True False  True  True False False  True  True  True
  True  True  True False  True  True False  True  True False  True  True
  True False False  True  True  True  True False False False False  True
  True False  True  True  True  True False False  True  True  True  True
 False False False  True False  True  True  True  True False  True  True
 False  True False False False  True  True False  True False False False
  True  True  True False  True  True  True  True  True  True False False
 False False  True  True  True False  True False  True False False False
 False False False  True False False False  True False  True  True  True
  True False False False  True False False  True False  True  True  True
 False  True  True  True False False  True  True False False  True  True
  True  True False False  True False  True  True  True False  True False
 False False  True  True False  True False False False  True False  True
  True False False  True False False  True False  True  True  True  True
 False  True  True  True  True False  True  True False False  True  True
  True  True False  True  True False  True False False False  True False
  True  True False False  True  True False  True  True  True  True  True
  True False  True  True False  True  True  True False False]
 Total Correct : 199
 Accuracy : 0.6180124223602484
                   precision    recall  f1-score   support

     Ariel Sharon       0.00      0.00      0.00        13
     Colin Powell       0.73      0.62      0.67        60
  Donald Rumsfeld       0.57      0.15      0.24        27
    George W Bush       0.62      0.90      0.74       146
Gerhard Schroeder       0.43      0.36      0.39        25
      Hugo Chavez       0.67      0.53      0.59        15
       Tony Blair       0.47      0.25      0.33        36

         accuracy                           0.62       322
        macro avg       0.50      0.40      0.42       322
     weighted avg       0.58      0.62      0.57       322
