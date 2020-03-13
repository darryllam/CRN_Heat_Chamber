/*Kurt Yesilcimen January 10th 2018
CRN Lab Project M2PC
*****Program info*******
This is a arduino code for using a normal PID with a gap 2C to control a heat system. This program
is setup to be used with 3 NTC 100kohm Thermistors and a solidstate relay. The Thermistors are then
averaged using a averaging formula.
Email complaints to Bryn Crawford [bryn.crawford@ubc.ca]
*****Hardware setup*****
SSR digital control pin: 3
Thermistor1 Analog Pin: 0
Thermistor2 Analog Pin: 1
Thermistor3 Analog Pin: 2
*/

//Headers
#include <PID_v1.h>
#include <SD.h>
#include <Wire.h>

#define ECHO_TO_SERIAL 1
#define WAIT_TO_START 0
#define LOG_INTERVAL  1000

//Program Variables
int SSR_Pin = 3; //SSR Pin, refer to hardware setup 
double Setpoint = 60; //Desired final temperature in celcius

//PID Variables
double Input, Output;
double aggKp = 8, aggKi = 1, aggKd = 3;
double consKp = 4, consKi = 0.2, consKd = 1;
PID myPID(&Input, &Output, &Setpoint, consKp, consKi, consKd, DIRECT);

//tempread variables
int ThermistorPin1 = 0; //pinout for thermistor1
int ThermistorPin2 = 1; //pinout for thermistor2
int ThermistorPin3 = 2; //pinout for thermistor3
int Vo;
int Vo1;
int Vo2;
int Vo3;
float R1 = 100000; //resistor1 value
float logR21, logR22, logR23, R21, R22, R23, T, T1, T2, T3, Tc, Tc1, Tc2, Tc3, Tf, Tf1, Tf2, Tf3, Ts;
float c1 = 2.418742546e-03, c2 = -0.8457770293e-04, c3 = 12.58925122e-07; //adjustable constant values

int ExtThermistorPin1 = 3;
int ExtVo1;
float ExtR21, logExtR21, ExtT1;

unsigned long StartTime;
File logfile

void setup() {
  Serial.begin(9600);
  //initialize varibles for PID
  Input = Tc;
  //turn the PID on
  myPID.SetMode(AUTOMATIC);
  pinmode(10, OUTPUT);
  if (!SD.begin(chipSelect)){
    Serial.println("Card failed, or not present");
    return;
  } else{
    Serial.println("Card initialized";
  }

  // create a new file
  char filename[] = "run00.CSV";
  for (uint8_t i = 0; i < 100; i++) {
    filename[6] = i/10 + '0';
    filename[7] = i%10 + '0';
    if (! SD.exists(filename)) {
      // only open a new file if it doesn't exist
      logfile = SD.open(filename, FILE_WRITE); 
      break;  // leave the loop!
    }
  }
  
  if (! logfile) {
    error("couldnt create file");
  }
  
  Serial.print("Logging to: ");
  Serial.println(filename);
  logfile.println(ExtTemp1,Time,);
  startTime = millis()/1000;
}

void loop() {
  read_temps();
  Input = Tc3;
  double gap = abs(Setpoint - Input); //distance away from setpoint
  if (gap < 2)
  {
    //Fine tuning temperature constants
    myPID.SetTunings(consKp, consKi, consKd);
  }
  else
  {
    //Agressive tuning temperature constants
    myPID.SetTunings(aggKp, aggKi, aggKd);
  }
  myPID.Compute();
  Output = Output*0.33;
  analogWrite(SSR_Pin, Output);
  //print pwm signal
  Serial.print(" | PWM= ");
  Serial.println(Output);
  delay(LOG_INTERVAL);
}

void read_part_temps(){
  ExtVo1 = analogRead(ExtThermistorPin1);
  ExtR21 = R1/ ((1023.0 / (floatExt)Vo1) - 1.0)
  logExtR21 = log(ExtR21);
  ExtT1 = (1.0 / (c1 + c2 * logExtR21 + c3 * logExtR21 * logExtR21 * logExtR21)) - 273.15;
  Serial.print("Ext T1: ");
  Serial.print(ExtT1);
  Serial.println(" C");
  logfile.print(ExtT1);
  logfile.print(", "); 
  logfile.print(millis/1000 - StartTime);
}

void read_temps() {
  Vo1 = analogRead(ThermistorPin1);
  Vo2 = analogRead(ThermistorPin2);
  Vo3 = analogRead(ThermistorPin3);
  //Thermistor1
  R21 = R1 / ((1023.0 / (float)Vo1) - 1.0);
  logR21 = log(R21);
  T1 = (1.0 / (c1 + c2 * logR21 + c3 * logR21 * logR21 * logR21));
  Tc1 = T1 - 273.15;
  Tf1 = (Tc1 * 9.0) / 5.0 + 32.0;
  //Thermistor2
  R22 = R1 / ((1023.0 / (float)Vo2) - 1.0);
  logR22 = log(R22);
  T2 = (1.0 / (c1 + c2 * logR22 + c3 * logR22 * logR22 * logR22));
  Tc2 = T2 - 273.15;
  Tf2 = (Tc2 * 9.0) / 5.0 + 32.0;
  //Thermistor3
  R23 = R1 / ((1023.0 / (float)Vo3) - 1.0);
  logR23 = log(R23);
  T3 = (1.0 / (c1 + c2 * logR23 + c3 * logR23 * logR23 * logR23));
  Tc3 = T3 - 273.15;
  Tf3 = (Tc3 * 9.0) / 5.0 + 32.0;
  //Thermistor Average
  T = (T1 + T2 + T3) / 3;
  Tc = T - 273.15;
  Tf = (Tc * 9.0) / 5.0 + 32.0;

  Serial.print("TemperatureAve: ");
  Serial.print(Tf);
  Serial.println(" F");
  Serial.print("TemperatureAve: ");
  Serial.print(Tc);
  Serial.println(" C");
  Serial.print("Temperature1: ");
  Serial.print(Tf1);
  Serial.println(" F");
  Serial.print("Temperature1: ");
  Serial.print(Tc1);
  Serial.println(" C");
  Serial.print("Temperature2: ");
  Serial.print(Tf2);
  Serial.println(" F");
  Serial.print("Temperature2: ");
  Serial.print(Tc2);
  Serial.println(" C");
  Serial.print("Temperature3: ");
  Serial.print(Tf3);
  Serial.println(" F");
  Serial.print("Temperature3: ");
  Serial.print(Tc3);
  Serial.println(" C");
}

