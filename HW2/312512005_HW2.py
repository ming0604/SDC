import numpy as np

def generate_weather_sequence(tran_matrix, initial, days):
    
    weather_sequence = [initial]
    for i in range(days-1):
        current = weather_sequence[-1]
        next = np.random.choice(len(tran_matrix), p=tran_matrix[current])
        weather_sequence.append(next)
    
    #print weather_sequence
    weather = ['sunny', 'cloudy', 'rainy']
    print("randomly generate weather sequences: ")
    for i in range(len(weather_sequence)):
        if weather_sequence[i]==0:
            print("Day {:d} : {:s} ,".format(i+1, weather[0]),end='')
            print(" the probability of tomorrow\'s weather will be: ", tran_matrix[0])
        elif weather_sequence[i]==1:
            print("Day {:d} : {:s} ,".format(i+1, weather[1]),end='')
            print(" the probability of tomorrow\'s weather will be: ", tran_matrix[1])
        elif weather_sequence[i]==2:
            print("Day {:d} : {:s} ,".format(i+1, weather[2]),end='')
            print(" the probability of tomorrow\'s weather will be: ", tran_matrix[2])
            
def stationary_distribution(tran_matrix, initial, days):
    if initial==0:
        weather_prob=np.array([1,0,0])
    elif initial==1:
        weather_prob=np.array([0,1,0])
    else:
        weather_prob=np.array([0,0,1])

    #count the probability of weather by Markov chain
    for i in range(days):
        weather_prob = np.dot(weather_prob, tran_matrix)
        print("The probability of the weather: ", weather_prob)

def main():
    weather_transition_matrix = np.array([   
        [0.8, 0.2, 0],
        [0.4, 0.4, 0.2],
        [0.2, 0.6, 0.2]
    ])

    num_days = int(input('Number of days you want to generate the weather sequence \n'))
    first_day_weather = int(input('Today\'s weather is? 1.sunny, 2.cloudy, 3.rainy\n'))



    #generate weather sequence based on given probabilities table
    generate_weather_sequence(weather_transition_matrix,first_day_weather-1,num_days)
    print("-" * 40)
    #count stationary weather distributions
    print('The probability of the stationary distributions of the weather:')
    stationary_distribution(weather_transition_matrix, first_day_weather-1 ,num_days)

if __name__ == '__main__':
    main()