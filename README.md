# Neural-Cryptography

The finalized code for the project is structured in the `asymmetric-encryption` folder.

The `EllipticCurve.py` file contains the code for generating the elliptic curve key pair. In the project, five different elliptic curves are used, and in this file the curve is chosen.

The `networks.py` file contains the code for creating the neural networks, Alice, Bob and Eve. The ABE-model and the Eve model, used for training, is also created in this file. The loss values is calculated and the optimizer and learning rate is set.

The `training.py` files contains the training of the ABE-model and Eve model. The number of epochs, batch size and Eve cycles is set in this file. The loss values are saved in a csv file structured as `'curve'/'Evecycles'cycle`. The files are saved as `test-'iteration-number'.csv`. The figures of each training is also saved in the folder `'curve'/'Evecycles'cycle/figures` as `result-'iteration-number'.png`, but these were only used during testing and not for the final report. The decryption accuracy of Bob and Eve are calculated after the training and stored in the file `result.txt`.

The `plot_between.py` file is used to generate the result figures used in the report. The figures created is the result of five iterations of the same curve and number of Eve cycles, and it uses the csv files from the folder `'curve'/'Evecycles'cycle`. The figures are stored in the `figures` folder.

The `average_loss.py` file is used to calculate the loss value of the ABE-model, Bob and Eve after training. It calculates the average of the last loss value after five iterations with the same curve and number of Eve cycles. It uses the csv files from the folder `'curve'/'Evecycles'cycle`. 

## Run the program
To train the neural network, select the preferred curve in `EllipticCurve.py` and:
```cd  asymmetric-encryption```
```python training.py```