from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
import numpy as np

# Set the elliptic curve
# Curves used in the project:
# curve = ec.SECP224R1()
# curve = ec.SECP256R1()
# curve = ec.SECP256K1()
# curve = ec.SECP384R1()
curve = ec.SECP521R1()

def generate_key_pair(batch_size):
    # Generate ECC private key
    pr_arr = np.empty((batch_size, 2304))
    pu_arr = np.empty((batch_size, 1720))
    for i in range(batch_size):
        private_key = ec.generate_private_key(
            curve, default_backend())
        # Derive the associated public key
        public_key = private_key.public_key()
        pr_arr[i], pu_arr[i] = convert_key_to_pem(
            batch_size, private_key, public_key)
    return pr_arr, pu_arr


def convert_key_to_pem(batch_size, private_key, public_key):
    # Convert keys to PEM format
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    ).decode()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    return convert_key_to_bit(batch_size, private_pem), convert_key_to_bit(batch_size, public_pem)


def convert_key_to_bit(batch_size, pem):
    # Convert PEM string to a bit string
    bits = ''.join([format(ord(c), '08b') for c in pem])
    arr = np.array([int(bit) for bit in bits])
    return arr


if __name__ == "__main__":
    pr_arr, pu_arr = generate_key_pair(1)
    print(pr_arr.shape)
    print(pu_arr.shape)