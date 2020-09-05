# ''' /**************************************************
#      * Set servos to start position.
#      * This is the position where the movement starts.
#      *************************************************/'''
#     writeAllServos(0);
# #ifdef DEBUG
#     for (uint8_t i = 0; i <= sServoArrayMaxIndex; ++i) {
#         sServoArray[i]->print(&Serial);
#     }
# #endif

# #if defined (SP)
#     Serial.print(F("Free Ram/Stack[bytes]="));
#     Serial.println(getFreeRam());
# #endif
# '''    // Wait for servos to reach start position.
# '''    delay(500);
# }

# void loop() {
# #ifdef INFO
#     Serial.print(F("Move all to 180 degree with 20 degree per second with "));  # 왜 180도를 20도/s 씩 움직이지? 한번에 움직이면 안되나?
#     Serial.print((180 * (1000L / 20)) / (sServoArrayMaxIndex + 1));
#     Serial.println(F(" ms delay"));
# #endif
#     setSpeedForAllServos(20);  '''// This speed is taken if no further speed argument is given.'''
#     for (uint8_t i = 0; i <= sServoArrayMaxIndex; ++i) {
#         sServoArray[i]->startEaseTo(180);
#      '''   /*
#          * Choose delay so that the last servo starts when the first is about to end
#          */'''
#         delay((180 * (1000L / 20)) / (sServoArrayMaxIndex + 1));
#     }
#     delay(1000);

# '''    // Now move back
# '''#ifdef INFO
#     Serial.println(F("Move all back to 0 degree with 20 degree per second"));
# #endif
#     for (uint8_t i = 0; i <= sServoArrayMaxIndex; ++i) {
#         sServoArray[i]->startEaseTo(0);
# #ifdef DEBUG
#         Serial.print(F("Start i="));
#         Serial.println(i);
# #endif
#        ''' /*
#          * Choose delay so that the last servo starts when the first is about to end
#          */'''
#         delay((180 * (1000L / 20)) / (sServoArrayMaxIndex + 1));
#     }

#     delay(1000);
# }

# '''/*
#  * Check if I2C communication is possible. If not, we will wait forever at endTransmission.
#  * 0x40 is default PCA9685 address
#  * @return true if error happened, i.e. device is not attached at this address.
#  */'''
# bool checkI2CConnection(uint8_t aI2CAddress) {
#     bool tRetValue = false;
#     Serial.print(F("Try to communicate with I2C device at address=0x"));
#     Serial.println(aI2CAddress, HEX);
#     Serial.flush();

#     Wire.beginTransmission(aI2CAddress);
#     uint8_t tWireReturnCode = Wire.endTransmission(true);
#     if (tWireReturnCode == 0) {
#         Serial.print(F("Found"));
#     } else {
#         Serial.print(F("Error code="));
#         Serial.print(tWireReturnCode);
#         Serial.print(F(". Communication with I2C was successful, but found no"));
#         tRetValue = true;
#     }
#     Serial.print(F(" I2C device attached at address: 0x"));
#     Serial.println(aI2CAddress, HEX);
#     return tRetValue;
# }

# '''/*
#  * Get the 16 ServoEasing objects for the PCA9685 expander
#  * The attach() function inserts them in the sServoArray[] array.
#  */'''
# void getAndAttach16ServosToPCA9685Expander(uint8_t aPCA9685I2CAddress) {
#     ServoEasing * tServoEasingObjectPtr;

#     Serial.print(F("Get ServoEasing objects and attach servos to PCA9685 expander at address=0x"));
#     Serial.println(aPCA9685I2CAddress, HEX);
#     for (uint8_t i = 0; i < PCA9685_MAX_CHANNELS; ++i) {
# #if defined(ARDUINO_SAM_DUE)
#         tServoEasingObjectPtr= new ServoEasing(aPCA9685I2CAddress, &Wire1);
# #else
#         tServoEasingObjectPtr = new ServoEasing(aPCA9685I2CAddress, &Wire);
# #endif
#         if (tServoEasingObjectPtr->attach(i) == INVALID_SERVO) {
#             Serial.print(F("Address=0x"));
#             Serial.print(aPCA9685I2CAddress, HEX);
#             Serial.print(F(" i="));
#             Serial.print(i);
#             Serial.println(
#                     F(
#                             " Error attaching servo - maybe MAX_EASING_SERVOS=" STR(MAX_EASING_SERVOS) " is to small to hold all servos"));

#         }
#     }
# }

# #if defined (SP)
# '''/*
#  * Get amount of free RAM = Stack - Heap
#  */'''
# uint16_t getFreeRam(void) {
#     extern unsigned int __heap_start;
#     extern void *__brkval;

#     uint16_t tFreeRamBytes;

#     if (__brkval == 0) {
#         tFreeRamBytes = SP - (int) &__heap_start;
#     } else {
#         tFreeRamBytes = SP - (int) __brkval;
#     }
#     return (tFreeRamBytes);
# }

# #endif