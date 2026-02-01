import subprocess
import sys
from flask import jsonify

# Helper function that helps to quickly get all the expected answers (from test cases that you come up with)

def main():
    run_l1_p1()
    # run_generic()
    # run_l3_c()

def run_generic():
    try:
        file_path = sys.argv[1]
        test_cases = ["1,1,1,2,3,4,4", "2,3,3,3,4,4,4,5,5", "1,2,3,4,5,6", "1,1,1,1,1,1,1", "1,3,5,6,7,7,7"]
        
        for case in test_cases:
            result = subprocess.run(['python3', file_path, case], capture_output=True, text=True)
            output = result.stdout
            print(output)

    except Exception as e:
        print("Error!")

def run_l1_p1():
    try:
        file_path = sys.argv[1]
        test_cases = ['1,2,3,4,5,6,7', '7,6,5,4,3,2,1', '1,3,5,1,3,5', '2,4,6,2,4,2', '2,3,5,1,4,6']
        
        for case in test_cases:
            result = subprocess.run(['python3', file_path, case], capture_output=True, text=True)
            output = result.stdout
            print(output)

    except Exception as e:
        print("Error!")
    
def run_l3_c():
    try:
        file_path = sys.argv[1]

        test_cases = ["50,20,60,10,30,55,80-N,L,R,LL,LR,RR,LLL", "1,2,3,4,5,6-N,L,LL,LLL,LLLL,LLLLL", "123-N", "1,2,3,5,8,13,21-N,L,R,LL,LR,RR,RL", "43,17,23,57,89,101,0,133-N,L,R,LL,LR,RL,LLL,LLR"]

        answers = ["3", "5", "0", "2", "3"]

        for i in range(len(test_cases)):
            result = subprocess.run(['python3', file_path, test_cases[i]], capture_output=True, text=True)
            output = result.stdout
            print(output)

    except Exception as e:
        print("Error!")

if __name__ == '__main__':
    main()