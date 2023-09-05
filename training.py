from networks import alice, bob, abemodel, m_train, m_bits, k_bits, evemodel
import numpy as np
import matplotlib.pyplot as plt
import sys

n_epochs = 20
batch_size = 512
n_batches = m_train // batch_size
abecycles = 1
evecycles = 2

epoch = 0
while epoch < n_epochs:
    evelosses = []
    boblosses = []
    abelosses = []
    for iteration in range(n_batches):
        # Train the A-B+E network
        alice.trainable = True
        for cycle in range(abecycles):
            # Select a random batch of messages, and a random batch of keys
            m_batch = np.random.randint(
                0, 2, m_bits * batch_size).reshape(batch_size, m_bits)
            k_batch = np.random.randint(
                0, 2, k_bits * batch_size).reshape(batch_size, k_bits)
            loss = abemodel.train_on_batch([m_batch, k_batch, k_batch], None)

        abelosses.append(loss)
        abeavg = np.mean(abelosses)

        # Evaluate Bob's ability to decrypt a message
        m_enc = alice.predict([m_batch, k_batch])
        m_dec = bob.predict([m_enc, k_batch])
        loss = np.mean(np.sum(np.abs(m_batch - m_dec), axis=-1))
        boblosses.append(loss)
        bobavg = np.mean(boblosses)

        # Train the EVE network
        alice.trainable = False
        for cycle in range(evecycles):
            m_batch = np.random.randint(
                0, 2, m_bits * batch_size).reshape(batch_size, m_bits)
            k_batch = np.random.randint(
                0, 2, k_bits * batch_size).reshape(batch_size, k_bits)
            loss = evemodel.train_on_batch([m_batch, k_batch], None)
        evelosses.append(loss)
        eveavg = np.mean(evelosses)

        if iteration % max(1, (n_batches // 100)) == 0:
            print("\rEpoch {:3}: {:3}% | abe: {:2.3f} | eve: {:2.3f} | bob: {:2.3f}".format(
                epoch, 100 * iteration // n_batches, abeavg, eveavg, bobavg), end="")
            sys.stdout.flush()

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
