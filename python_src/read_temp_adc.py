import time
import board
import busio
import math
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
file_name = 'run0.txt'

c1,c2,c3 = 0.6764629190e-03, 2.230798167e-04, 0.7159342899e-07;
adc_max = 2<<16 - 1
input_ports = 4
Vdd = 3.3
sample_rate = .5
sample_rate = sample_rate - .0777
voltage_data=[0]*input_ports
adc_value_data = [0]*input_ports
temps = [0]*input_ports
R_ref = [100e3] * input_ports


# Create the I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

# Create the ADC object using the I2C bus
ads = ADS.ADS1115(i2c)
# you can specify an I2C adress instead of the default 0x48
#ads = ADS.ADS1115(i2c, address=0x49)

# Create single-ended input on channel 0
chan0 = AnalogIn(ads, ADS.P0)
# Create single-ended input on channel 1
chan1 = AnalogIn(ads, ADS.P1)
# Create single-ended input on channel 2
chan2 = AnalogIn(ads, ADS.P2)
# Create single-ended input on channel 3
chan3 = AnalogIn(ads, ADS.P3)

ads.gain = 1 #measure voltages +/-4.096V

start_time = time.time()
with open(file_name, "w") as text_file:
    while True: 
        voltage_data[0] = chan0.voltage
        voltage_data[1] = chan1.voltage
        voltage_data[2] = chan2.voltage
        voltage_data[3] = chan3.voltage
        adc_value_data[0] = chan0.value
        time.sleep(sample_rate/(input_ports))
        adc_value_data[1] = chan1.value
        time.sleep(sample_rate/(input_ports))
        adc_value_data[2] = chan2.value
        time.sleep(sample_rate/(input_ports))
        adc_value_data[3] = chan3.value
        time.sleep(sample_rate/(input_ports))    
        write_data = ''

        for i in range(0,input_ports):
            #print("Channel "+str(i)+" Value: "+str(adc_value_data[i])+", Voltage: "+str(voltage_data[i]))
            R_therm = R_ref[i] / ((Vdd / (Vdd - voltage_data[i])) - 1.0)
            logR_threm = math.log(R_therm)
            temps[i] = (1.0 / (c1 + c2 * logR_threm + c3 * logR_threm * logR_threm * logR_threm))
            temps[i] = temps[i] - 273.15
            print("{0:0.4f}C | ".format(temps[i]), end='')
            write_data = write_data + str("{0:0.4f}".format(temps[i])) + ","
        
        write_data = write_data + str("{0:0.4f}".format(time.time()-start_time)) + "\n"
        text_file.write(write_data)
        print("Time: "+str(time.time() - start_time))
