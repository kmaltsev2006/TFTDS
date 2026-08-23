#include <boost/asio.hpp>
#include <boost/asio/error.hpp>
#include <boost/asio/experimental/concurrent_channel.hpp>
#include <boost/asio/thread_pool.hpp>

#include <cassert>
#include <chrono>
#include <iostream> // TODO: remove
#include <memory>
#include <thread>

// TODO: fix overhead with shared_ptr copying

static const uint kThreadCount = std::thread::hardware_concurrency();

template <typename T> using Matrix = std::vector<std::vector<T>>;

using boost_error_code = boost::system::error_code;

static const uint32_t kBlockSize = 1;

class Accumulator {
    template <typename ExecutorOrSignature, typename... Signatures>
    using channel =
        boost::asio::experimental::concurrent_channel<ExecutorOrSignature, Signatures...>;

  public:
    Accumulator() = default;

    ~Accumulator() { assert(_wait_called); }

    // Non-copyable
    Accumulator(const Accumulator &other) = delete;
    Accumulator &operator=(const Accumulator &other) = delete;

    // Non-movable
    Accumulator(Accumulator &&other) = delete;
    Accumulator &operator=(Accumulator &&other) = delete;

    void wait() {
        _workers_pool.wait();
        _ch_pool.wait();
        _wait_called = true;
    }

    auto &wPool() { return _workers_pool; }

    template <typename T> auto getTnCh(const size_t n, const size_t bl_cnt) {
        std::shared_ptr ch_ptr{
            std::make_shared<channel<void(boost_error_code)>>(_ch_pool.get_executor(), bl_cnt)};

        std::shared_ptr t_ptr{std::make_shared<Matrix<T>>(n, std::vector<T>(n))};

        boost::asio::post(_ch_pool, [ch_ptr, t_ptr, bl_cnt] {
            std::shared_ptr calculated_bl_cnt_ptr{std::make_shared<std::atomic<size_t>>(0)};

            for (size_t bl = 0; bl < bl_cnt; ++bl) {
                ch_ptr->async_receive([t_ptr, calculated_bl_cnt_ptr, bl_cnt](boost_error_code ec) {
                    // using namespace std::literals::chrono_literals;

                    assert(!ec);

                    if (calculated_bl_cnt_ptr->fetch_add(1) == bl_cnt - 1) {
                        auto &t = *t_ptr;
                        // std::for_each(t.begin(), t.end(), [](const auto &row) {
                        //     std::for_each(row.begin(), row.end(),
                        //                   [](const auto &el) { std::cout << el << " "; });
                        //     std::cout << std::endl;
                        // });
                    }
                });
            }
        });

        return std::make_pair(t_ptr, ch_ptr);
    }

  private:
    boost::asio::thread_pool _ch_pool{kThreadCount};
    boost::asio::thread_pool _workers_pool{kThreadCount};
    bool _wait_called{false};
};

Accumulator acc;

template <typename T> void transpose(const Matrix<T> &m) {
    // using namespace std::literals::chrono_literals;

    static const auto transponseBlock = [](auto &t, const auto &m, const size_t bi, const size_t bj) {
        for (size_t i = 0; i < kBlockSize; ++i)
            for (size_t j = 0; j < kBlockSize; ++j)
                t[bi * kBlockSize + i][bj * kBlockSize + j] =
                    m[bj * kBlockSize + j][bi * kBlockSize + i];
    };

    const size_t n = m.size();

    assert(n > 0 && n == m[0].size());
    assert(n % kBlockSize == 0);

    size_t bl_cnt = (n / kBlockSize) * (n / kBlockSize);

    auto [t_ptr, ch_ptr] = acc.getTnCh<T>(n, bl_cnt);

    for (size_t bi = 0; bi < n / kBlockSize; ++bi) {
        for (size_t bj = 0; bj < n / kBlockSize; ++bj) {

            boost::asio::post(acc.wPool(), [t_ptr, ch_ptr, &m, bi, bj] {
                auto& t = *t_ptr;

                transponseBlock(t, m, bi, bj);

                ch_ptr->async_send({}, [](boost_error_code ec) { assert(!ec); });
            });
        }
    }
}

template <typename T> void transpose0(const Matrix<T> &m) {
    // using namespace std::literals::chrono_literals;

    static const auto transponseBlock = [](auto &t, const auto &m, const size_t bi, const size_t bj) {
        for (size_t i = 0; i < kBlockSize; ++i)
            for (size_t j = 0; j < kBlockSize; ++j)
                t[bi * kBlockSize + i][bj * kBlockSize + j] =
                    m[bj * kBlockSize + j][bi * kBlockSize + i];
    };
    
    const size_t n = m.size();

    assert(n > 0 && n == m[0].size());
    assert(n % kBlockSize == 0);

    Matrix<T> t(n, std::vector<T>(n));

    for (size_t bi = 0; bi < n / kBlockSize; ++bi) {
        for (size_t bj = 0; bj < n / kBlockSize; ++bj) {
            transponseBlock(t, m, bi, bj);
        }
    }
}

int main() {
    int n{10};
    std::vector<std::vector<int>> a(n, std::vector<int>(n));
    int k{0};
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i][j] = ++k;
        }
    }

    for (int i = 0; i < 100; ++i) {
        transpose(a);
    }

    acc.wait();
}