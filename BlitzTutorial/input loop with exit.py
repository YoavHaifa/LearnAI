# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:29:41 2019

@author: yoavb
"""

while True:
   answer = input('Do you want to continue?:')
   if answer.lower().startswith("y"):
      print("ok, carry on then")
   elif answer.lower().startswith("n"):
      print("sayonara, Robocop")
      exit()
      print("after exit")
      raise SystemExit
      print("after SystemExit")
     
      
      