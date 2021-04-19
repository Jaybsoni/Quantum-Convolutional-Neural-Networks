import numpy as np

class Data():
    def __init__(self):
        csv = np.genfromtxt('even_mnist.csv', delimiter=" ")

        # Splits data into training and testing sets are requested by assignment
        self.x_train = np.float32(np.interp(csv[:-3000, :-1], (0, 255), (0, 1)))
        self.x_test = np.float32(np.interp(csv[-3000:, :-1], (0, 255), (0, 1)))

        # Extract the answers for the training and testing data
        y_train = csv[:-3000, -1:]
        y_test = csv[-3000:, -1:]

        self.x_train, self.y_train = self.scrape_numbers(self.x_train, y_train)
        self.x_test, self.y_test = self.scrape_numbers(self.x_test, y_test)


    # Bit slow, but for last minute it does the trick
    def scrape_numbers(self, x, y):
        for removed_number in [0, 2, 8]:
            print(f"Scraping {removed_number} from data")
            elements = np.where(y == removed_number)[0]
            reversed_elements = elements[::-1]
            for i in reversed_elements:
                x = np.delete(x, i, 0)
                y = np.delete(y, i, 0)

        print(np.where(y == 0)[0])
        print(np.where(y == 2)[0])
        print(np.where(y == 8)[0])

        return x, y



if __name__ == '__main__':
    d = Data()