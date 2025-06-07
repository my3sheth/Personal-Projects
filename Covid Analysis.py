import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

print('-------------------------------------------------------------------------')
print('\t\t\t COVID ANALYSIS \t\t\t')
print('-------------------------------------------------------------------------')

engine = create_engine('mysql+pymysql://root:abcde@localhost:3306/MYDB')

df1 = pd.read_sql('SELECT * FROM ACTIVE_CASES', engine)
d1 = pd.DataFrame(df1)

df2 = pd.read_sql('SELECT * FROM CONFIRMED_CASES', engine)
d2 = pd.DataFrame(df2)

df3 = pd.read_sql('SELECT * FROM DEATHS', engine)
d3 = pd.DataFrame(df3)

df4 = pd.read_sql('SELECT * FROM RECOVERIES', engine)
d4 = pd.DataFrame(df4)

z1 = d1['END_I'].sum()
z2 = d2['END_I'].sum()
z3 = d3['END_I'].sum()
z4 = d4['END_I'].sum()

z5 = d1['END_II'].sum()
z6 = d2['END_II'].sum()
z7 = d3['END_II'].sum()
z8 = d4['END_II'].sum()

x = ['ACTIVE', 'CONFIRMED', 'DEATHS', 'RECOVERIES']

a = 0
while a != -1:
    print('\nMODULES AVAILABLE TO VIEW:\n')
    print('1. FIRST AND SECOND WAVE COMPARISON')
    print('2. ACTIVE CASES')
    print('3. CONFIRMED CASES')
    print('4. DEATHS')
    print('5. RECOVERIES')
    print('6. GENDER-WISE IMPACT')
    print()

    a = int(input('Choose module to view:'))

    if a == 1:
        plt.ticklabel_format(style='plain')
        plt.plot(x, [z5, z6, z7, z8], marker='*', markersize=10, linewidth=2, color='violet')
        plt.plot(x, [z1, z2, z3, z4], marker='*', markersize=10, linewidth=2, color='blue')
        plt.legend(['Second Wave', 'First Wave'])
        plt.ylabel('NUMBERS')
        plt.title('FIRST WAVE AND SECOND WAVE COMPARISON')
        plt.show()
    
    elif a == 2:
        print('\n\t\t\t ACTIVE CASES\n', df1)
        print('\t\t\t ____________\t\t\t\n')
        b = input('Do you want to view the graph?(YES/NO):')
        if b.lower() == 'yes':
            d1.plot(kind='barh', x='CITY', color=['b', 'r', 'm', 'gold'])
            plt.legend()
            plt.title('ACTIVE CASES')
            plt.ylabel('CITIES')
            plt.xlabel('NO. OF CASES')
            plt.show()
            print()
        else:
            exit

    elif a == 3:
        print('\t\t\t CONFIRMED CASES\n', df2)
        print('\t\t\t ____________\t\t\t\n')
        b = input('Do you want to view the graph?(YES/NO):')
        if b.lower() == 'yes':
            d2.plot(kind='barh', x='CITY', color=['b', 'r', 'm', 'gold'])
            plt.legend()
            plt.title('CONFIRMED CASES')
            plt.ylabel('CITIES')
            plt.xlabel('NO. OF CASES')
            plt.show()
            print()
        else:
            exit

    elif a == 4:
        print('\t\t\t DEATHS\n', df3)
        print('\t\t\t ____________\t\t\t\n')
        b = input('Do you want to view the graph?(YES/NO):')
        if b.lower() == 'yes':
            d3.plot(kind='barh', x='CITY', color=['b', 'r', 'm', 'gold'])
            plt.legend()
            plt.title('DEATHS')
            plt.ylabel('CITIES')
            plt.xlabel('NO. OF CASES')
            plt.show()
            print()
        else:
            exit

    elif a == 5:
        print('\t\t\t RECOVERIES\n', df4)
        print('\t\t\t ____________\t\t\t\n')
        b = input('Do you want to view the graph?(YES/NO):')
        if b.lower() == 'yes':
            d4.plot(kind='barh', x='CITY', color=['b', 'r', 'm', 'gold'])
            plt.legend()
            plt.title('RECOVERIES')
            plt.ylabel('CITIES')
            plt.xlabel('NO. OF CASES')
            plt.show()
            print()
        else:
            exit

    elif a == 6:
        gender = ['Male', 'Female']
        n = [73797, 39063]
        color = ['Cyan', 'Red']
        plt.pie(n, labels=gender, colors=color, autopct='%.2f', explode=[0.1, 0.1], shadow=True)
        plt.title('Gender-wise Impact')
        plt.legend()
        plt.show()

    else:
        break

print('\nThank you for viewing our project!')
print('Let us fight against and sustain ourselves amidst these tough times by wearing a mask and following covid protocols, and be kinder to everyone around us.')
