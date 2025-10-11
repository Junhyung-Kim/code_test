#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;

// 간단한 계좌 클래스
class Account {
public:
    string accountNumber;
    int balance;

    Account(const string& accNum, int initialBalance)
        : accountNumber(accNum), balance(initialBalance) {}

    int getBalance() const {
        return balance;
    }

    void deposit(int amount) {
        balance += amount;
    }

    bool withdraw(int amount) {
        if (balance >= amount) {
            balance -= amount;
            return true;
        }
        return false;
    }
};

// ATM 컨트롤러 클래스
class ATMController {
private:
    unordered_map<string, string> pinDatabase; // 카드번호 -> PIN
    unordered_map<string, vector<Account>> accountsDatabase; // 카드번호 -> 계좌 리스트
    string currentCard;
    bool cardInserted = false;

public:
    ATMController() {
        pinDatabase["1111-1111-1111-1111"] = "1111";
        accountsDatabase["1111-1111-1111-1111"] = {
            Account("1111", 200), // (계좌이름, 계좌잔액)
            Account("2222", 200) 
        };
    }

    bool insertCard(const string& cardNumber) {
        if (pinDatabase.find(cardNumber) != pinDatabase.end()) {
            currentCard = cardNumber;
            cardInserted = true;
            return true;
        }
        return false;
    }

    bool enterPIN(const string& pin) {
        if (!cardInserted) return false;
        return pinDatabase[currentCard] == pin;
    }

    vector<string> listAccounts() {
        vector<string> accList;
        if (!cardInserted) return accList;
        for (auto &acc : accountsDatabase[currentCard]) {
            accList.push_back(acc.accountNumber);
        }
        return accList;
    }

    int checkBalance(int accountIndex) {
        if (!cardInserted) return -1;
        if (accountIndex < 0 || accountIndex >= accountsDatabase[currentCard].size()) return -1;
        return accountsDatabase[currentCard][accountIndex].getBalance();
    }

    bool deposit(int accountIndex, int amount) {
        if (!cardInserted) return false;
        if (accountIndex < 0 || accountIndex >= accountsDatabase[currentCard].size()) return false;
        accountsDatabase[currentCard][accountIndex].deposit(amount);
        return true;
    }

    bool withdraw(int accountIndex, int amount) {
        if (!cardInserted) return false;
        if (accountIndex < 0 || accountIndex >= accountsDatabase[currentCard].size()) return false;
        return accountsDatabase[currentCard][accountIndex].withdraw(amount);
    }

    void removeCard() {
        cardInserted = false;
        currentCard = "";
    }
};

// 테스트 코드
int main() {
    ATMController atm;

    // 카드 삽입
    if (!atm.insertCard("1111-1111-1111-1111")) {
        cout << "Invalid card!" << endl;
        return 1;
    }

    // PIN 입력
    if (!atm.enterPIN("1111")) {
        cout << "Incorrect PIN!" << endl;
        return 1;
    }

    // 계좌 선택
    vector<string> accounts = atm.listAccounts();
    cout << "Accounts:" << endl;
    for (int i = 0; i < accounts.size(); ++i) {
        cout << i << ": " << accounts[i]
             << " (Balance: $" << atm.checkBalance(i) << ")" << endl;
    }

    // 입금
    atm.deposit(0, 50);
    cout << "After deposit, account 0 balance: $" << atm.checkBalance(0) << endl;

    // 출금
    if (atm.withdraw(1, 150)) {
        cout << "Withdraw successful, account 1 balance: $" << atm.checkBalance(1) << endl;
    } else {
        cout << "Insufficient funds for withdrawal!" << endl;
    }

    // 카드 제거
    atm.removeCard();

    return 0;
}
