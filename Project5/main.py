import warnings
warnings.simplefilter('ignore')

from myAAS import ask_the_ensemble


# Please insert path to your text file containing the attacks
y = ask_the_ensemble(input_name = "AdversarialText.txt")

print(y) # X has also been saved to your directory where the code is located

output = ""
for item in y:
    output += "\n"+item

# Write the result to the File
file = open("AdversarialTestResults.txt","w")
file.write(output) 
file.close() 










