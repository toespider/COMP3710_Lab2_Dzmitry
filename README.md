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


Using device: Tesla T4
100%|██████████| 170M/170M [10:32<00:00, 270kB/s]

--- Running Single Inference Pass (Demonstration Req) ---
Inference step complete.
--- Running Single Training Epoch (Demonstration Req) ---
Single epoch complete in 45.61 seconds.

--- Starting Full Training for 30 Epochs ---
Epoch 05/30 | Test Accuracy: 71.32% | Elapsed: 99.7s

        macro avg       0.50      0.40      0.42       322
     weighted avg       0.58      0.62      0.57       322
