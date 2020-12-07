# LoRaWAN Localization using Data Rate Aware Fingerprinting Network  

Localization is an essential in any Internet-of-Things(IoT)  implementation  to  give  meaning  to  its  data.  Low  PowerWide  Area  Network  (LPWAN)  technology  such  as  LoRaWANenables  GPS-free  localization  through  its  long-long  range  com-munication  and  low  power  consumption.  Various  ranging  andfingerprinting techniques have been studied so far. However, theysuffer high positioning errors due to the Received Signal Strength(RSSI) variation. In this study, a Data Rate Aware FingerprintingNetwork for LoRaWAN localization is developed to mitigate RSSIvariation using the Spreading Factor SF information from sensornodes  that  preprocess  the  RSSI  out  of  range  data  to  improvethe  performance.  A  publicly  available  dataset  that  offers  signalstrength and SF of the sensor node is used and is split to train,validate,  and  test  the  neural  network  model.  A  reduced  versionof  the  dataset  is  also  utilized  wherein  messages  with  less  thanthree  receiving  base  station  are  discarded.  The  fingerprintingmodel  achieved  a  mean  positioning  error  of  293.60  meters  onthe  publicly  available  dataset  and  225.57  meters  in  the  reducedversion.
