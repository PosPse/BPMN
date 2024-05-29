#include <iostream>
int main()
{
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            if (j == 2)
            {
                i = j;
                std::cout << i << " " << j << std::endl;
            }
        }
    }
}