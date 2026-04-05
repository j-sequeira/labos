import smbus
AS5600_adress = 0x36 # Default device I2C address
bus = smbus.SMBus(1)

bus.write_i2c_block_data(AS5600_adress,0x08,[0xE0])

def ReadRawAngle(): # Read angle (0-360 represented as 0-4096)
    #bus.write_byte(AS5600_adress,0x0E)
    read_bytes = bus.read_i2c_block_data(AS5600_adress,0x0E,2)
    return (read_bytes[0]<<8) | read_bytes[1];

def ang(bit):
    return bit*360/4096

while True:
    print(ang(ReadRawAngle()))