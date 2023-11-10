from cryptography.hazmat.primitives.asymmetric import rsa
import numpy as np
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa


# def key_to_int_array(key):
#     if isinstance(key, rsa.RSAPrivateKey):
#         numbers = key.private_numbers()
#         bytes_val = numbers.d.to_bytes(  # Use the private exponent 'd' for private key
#             (numbers.d.bit_length() + 7) // 8, byteorder='big')
#     else:
#         numbers = key.public_numbers()
#         bytes_val = numbers.n.to_bytes(  # Use the modulus 'n' for public key
#             (numbers.n.bit_length() + 7) // 8, byteorder='big')

#     return [int(b) for b in bytes_val]


# def generate_key_pair_as_int():
#     private_key = rsa.generate_private_key(
#         public_exponent=65537,
#         key_size=2048
#     )
#     public_key = private_key.public_key()
#     return key_to_int_array(public_key), key_to_int_array(private_key)
def generate_key_pair():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    public_key = private_key.public_key()
    return (public_key), (private_key)


def generate_keys(batch_size):
    public_keys = np.empty((batch_size, 256))
    private_keys = np.empty((batch_size, 256))
    for i in range(batch_size):
        public, private = generate_key_pair()
        public_keys[i] = public
        private_keys[i] = private
    return public_keys, private_keys

# def generate_key_pair():
#     private_key = rsa.generate_private_key(
#         public_exponent=65537,
#         key_size=2048,
#     )
#     return private_key.public_key(), private_key


# def serialize_keys(public_key, private_key):
#     private = private_key.private_bytes(
#         encoding=serialization.Encoding.PEM,
#         format=serialization.PrivateFormat.TraditionalOpenSSL,
#         encryption_algorithm=serialization.NoEncryption()
#     )
#     public = public_key.public_bytes(
#         encoding=serialization.Encoding.PEM,
#         format=serialization.PublicFormat.SubjectPublicKeyInfo
#     )
#     return public, private


# def generate_keys(batch_size):
#     public_key = np.empty((batch_size, 1), dtype=object)
#     private_key = np.empty((batch_size, 1), dtype=object)
#     for i in range(batch_size):
#         pu, pr = generate_key_pair()
#         public, private = serialize_keys(pu, pr)
#         public_key[i] = public
#         private_key[i] = private
#     return np.array(public_key), np.array(private_key)

if __name__ == "__main__":
    # Generate keys
    publicKey, privateKey = rsa.newkeys(512)

    # Convert private key to PEM format
    private_pem = privateKey.save_pkcs1(format='PEM').decode()
    public_pem = publicKey.save_pkcs1(format='PEM').decode()

    # Convert PEM string to a bit string
    private_bits = ''.join([format(ord(c), '08b') for c in private_pem])
    private_arr = np.array([int(bit) for bit in private_bits])
    public_bits = ''.join([format(ord(c), '08b') for c in public_pem])
    public_arr = np.array([int(bit) for bit in public_bits])

    # The part if we want to add paddding and setting the length of the private key to a given size
    desired_length = 512 * 8  # 2048 bytes * 8 bits/byte
    while len(private_arr) < desired_length:
        private_arr = np.append(private_arr, 0)
    reconstructed_pem_bytes = bytearray()
    for i in range(0, len(private_arr), 8):
        byte_value = int(''.join(map(str, private_arr[i:i+8])), 2)
        reconstructed_pem_bytes.append(byte_value)

    # Convert the PEM bytes back to an RSA key
    reconstructed_private_key = rsa.PrivateKey.load_pkcs1(
        bytes(reconstructed_pem_bytes))

    # Encrypt a message
    message = "Hello Mina!"
    encryptedMessage = rsa.encrypt(message.encode(), publicKey)

    # Decrypt the message
    decryptedMessage = rsa.decrypt(
        encryptedMessage, reconstructed_private_key).decode()

    # ---
    pub, priv = generate_keys(5)
    print(pub.shape)
    print(priv.shape)
