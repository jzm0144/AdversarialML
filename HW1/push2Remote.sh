#!/bin/bash


echo "You Entered local address"
read filePath

echo "You Entered remote address"
read remote

#remote = "jzm0144@hopper.auburn.edu:/home/jzm0144/Janzaib_Playground/AdversairalML_Course"


scp $filePath $remote

echo "File Successfully Sent to your Hopper AdversarialML_Course directory"
