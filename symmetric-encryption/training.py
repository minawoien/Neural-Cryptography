from networks import alice, bob, eve, abemodel, m_train, m_bits, k_bits, evemodel
import numpy as np
import matplotlib.pyplot as plt
import sys

abelosses = []
boblosses = []
evelosses = []


n_epochs = 20
batch_size = 1000
n_batches = m_train // batch_size
abecycles = 1
evecycles = 1

epoch = 0
while epoch < n_epochs:
    evelosses0 = []
    boblosses0 = []
    abelosses0 = []
    for iteration in range(n_batches):
        # Train the A-B+E network
        alice.trainable = True
        for cycle in range(abecycles):
            # Select a random batch of messages, and a random batch of keys
            m_batch = np.random.randint(
                0, 2, m_bits * batch_size).reshape(batch_size, m_bits)
            k_batch = np.random.randint(
                0, 2, k_bits * batch_size).reshape(batch_size, k_bits)
            loss = abemodel.train_on_batch(
                [m_batch, k_batch, k_batch], None)

        abelosses0.append(loss)
        abelosses.append(loss)
        abeavg = np.mean(abelosses0)

        # Evaluate Bob's ability to decrypt a message
        m_enc = alice.predict([m_batch, k_batch])
        m_dec = bob.predict([m_enc, k_batch])
        loss = np.mean(np.sum(np.abs(m_batch - m_dec), axis=-1))
        boblosses0.append(loss)
        boblosses.append(loss)
        bobavg = np.mean(boblosses0)

        # Train the EVE network
        alice.trainable = False
        for cycle in range(evecycles):
            m_batch = np.random.randint(
                0, 2, m_bits * batch_size).reshape(batch_size, m_bits)
            k_batch = np.random.randint(
                0, 2, k_bits * batch_size).reshape(batch_size, k_bits)
            loss = evemodel.train_on_batch([m_batch, k_batch], None)
        evelosses0.append(loss)
        evelosses.append(loss)
        eveavg = np.mean(evelosses0)

        if iteration % max(1, (n_batches // 100)) == 0:
            print("\rEpoch {:3}: {:3}% | abe: {:2.3f} | eve: {:2.3f} | bob: {:2.3f}".format(
                epoch, 100 * iteration // n_batches, abeavg, eveavg, bobavg), end="")
            sys.stdout.flush()

    print()
    epoch += 1

print("Training complete.")
steps = -1

plt.figure(figsize=(7, 4))
plt.plot(abelosses[:steps], label='A-B')
plt.plot(evelosses[:steps], label='Eve')
plt.plot(boblosses[:steps], label='Bob')
plt.xlabel("Iterations", fontsize=13)
plt.ylabel("Loss", fontsize=13)
plt.legend(fontsize=13)

plt.show()


# Test the model
m_batch = np.random.randint(
    0, 2, m_bits * batch_size).reshape(batch_size, m_bits).astype('float32')
# guessing private key, input for eve should be cipertext and public key
k_batch = np.random.randint(
    0, 2, k_bits * batch_size).reshape(batch_size, k_bits).astype('float32')
cipher = alice.predict([m_batch, k_batch])

print(m_batch)  # original message

decrypted = bob.predict([cipher, k_batch])
print(decrypted)  # bob's attempt to decrypt
decrypted_bits = np.round(decrypted).astype(int)
print(decrypted_bits)

correct_bits = np.sum(decrypted_bits == m_batch)
total_bits = np.prod(decrypted_bits.shape)
accuracy = correct_bits / total_bits * 100

print(f"Number of correctly decrypted bits: {correct_bits}")
print(f"Total number of bits: {total_bits}")
print(f"Decryption accuracy: {accuracy}%")

eve_decrypted = eve.predict(cipher)
eve_decrypted_bits = np.round(eve_decrypted).astype(int)
print(eve_decrypted_bits)

correct_bits_eve = np.sum(eve_decrypted_bits == m_batch)
total_bits = np.prod(eve_decrypted_bits.shape)
accuracy_eve = correct_bits_eve / total_bits * 100

print(f"Number of correctly decrypted bits by Eve: {correct_bits_eve}")
print(f"Total number of bits: {total_bits}")
print(f"Decryption accuracy by Eve: {accuracy_eve}%")
