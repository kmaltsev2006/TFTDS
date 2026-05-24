#include <cassert>
#include <chrono>
#include <condition_variable>
#include <format>
#include <fstream>
#include <iostream>
#include <list>
#include <mutex>
#include <queue>
#include <random>
#include <syncstream>
#include <thread>
#include <utility>

#define DBG

static std::random_device rd;
static std::mt19937 rng{rd()};
static std::uniform_int_distribution<std::mt19937::result_type> dist{1, 10};

size_t thisThreadId() { return std::hash<std::thread::id>{}(std::this_thread::get_id()); }

template <typename ObjT>
class ObjPool;

class Printer {
    friend ObjPool<Printer>;

   public:
    uint64_t paper() const noexcept { return _paper; }

    /**
     * @brief Imitates printing one sheet of paper
     *
     * @return true if success
     * @return false if fail
     */
    bool print_one() {
        if (_paper == 0) {
            return false;
        }
        std::this_thread::sleep_for(_print_time_ms);
        --_paper;
#if defined(DBG)
        _f << std::format("{} | paper left: {}\n", thisThreadId(), _paper);
#endif
        return true;
    }

#if defined(DBG)
    ~Printer() { _f.close(); }
#endif

   private:
    uint64_t _paper{};
    std::chrono::milliseconds _print_time_ms{};
#if defined(DBG)
    std::ofstream _f{std::format("{}", static_cast<void*>(this))};
#endif

   private:
    explicit Printer(uint64_t paper, uint64_t print_time_ms = 50)
        : _paper(paper), _print_time_ms(print_time_ms) {}
};

template <typename ObjT>
class ObjPool {
    // For testing purpose
    friend uint64_t paperLeftSum(ObjPool<Printer>& obj_pool);
    friend int main();

    // template<typename T>
    // using seq_t = std::list<T>; // underling _objects queue container

    using obj_ptr_t = std::unique_ptr<ObjT>;

    class ObjPtrWrapper {
        friend ObjPool;

       public:
        explicit ObjPtrWrapper(obj_ptr_t&& obj_ptr, ObjPool& parrent)
            : _obj_ptr{std::forward<obj_ptr_t>(obj_ptr)}, _parrent{parrent} {}

        obj_ptr_t& operator->() { return _obj_ptr; }

        ~ObjPtrWrapper() { _parrent.release(std::move(_obj_ptr)); }

       private:
        obj_ptr_t _obj_ptr;
        ObjPool& _parrent;
    };

   public:
    template <typename... ObjArgs>
    explicit ObjPool(size_t n, ObjArgs&&... obj_args) : _size{n} {
        for (size_t i = 0; i < _size; ++i) {
            _objects.emplace_back(new ObjT(std::forward<ObjArgs>(obj_args)...));
        }
    }

    size_t size() const noexcept { return _size; }

    ObjPtrWrapper acquire() {
        std::unique_lock lk{_mutex};

        _cv.wait(lk, [this] { return !_objects.empty(); });

        auto obj_ptr = std::move(_objects.front());
        _objects.pop_front();
        return ObjPtrWrapper{std::move(obj_ptr), *this};
        ;
    }

   private:
    /*
     * A queue is used instead of a stack for "fair" resource usage.
     * In this implementation, a std::deque is used because a std::queue
     * cannot be iterated through for testing purposes. The deque can
     * be switched directly to a queue, a list, or any other container
     * of your choice depending on your goals.
     */
    std::deque<obj_ptr_t> _objects;  // guarded by _mutex
    std::mutex _mutex;               // guards _objects
    std::condition_variable _cv;
    const size_t _size;

   private:
    void release(obj_ptr_t&& obj_ptr) {
        std::unique_lock lk{_mutex};
        _objects.push_back(std::forward<obj_ptr_t>(obj_ptr));
        lk.unlock();

        _cv.notify_one();
    }
};

uint64_t paperLeftSum(ObjPool<Printer>& obj_pool) {
    return std::accumulate(obj_pool._objects.cbegin(), obj_pool._objects.cend(), uint64_t{},
                           [](uint64_t sum, const auto& rhs) { return sum + rhs->paper(); });
}

int main() {
    constexpr size_t printers = 10;
    constexpr uint64_t _paper_in_each = 1717;

    ObjPool<Printer> obj_pool{printers, _paper_in_each};

    auto worker_job = [&obj_pool](uint64_t papers_to_print) {
#if defined(DBG)
        std::osyncstream(std::cout)
            << std::format("{} | will try to print {}\n", thisThreadId(), papers_to_print);
#endif
        auto printer = obj_pool.acquire();
        while (papers_to_print--) {
            if (!printer->print_one()) {
                return;
            }
        }
    };

    uint64_t paper_to_print_sm{};
    {
        std::vector<std::jthread> workers;
        for (int i = 0; i < 100; ++i) {
            uint64_t papers_to_print = dist(rng);
            workers.emplace_back(worker_job, papers_to_print);
            paper_to_print_sm += papers_to_print;
        }
    }  // sync workers

    std::cout << std::format("papers at start: {}\npapers to print: {}\npapers left: {}\n",
                             printers * _paper_in_each, paper_to_print_sm, paperLeftSum(obj_pool));

    // Works only if papers at start >= papers to print
    assert(paper_to_print_sm == printers * _paper_in_each - paperLeftSum(obj_pool));

    std::cout << std::format("obj_pool expected size {}\nobj_pool real size {}\n", obj_pool.size(),
                             obj_pool._objects.size());
    
    assert(obj_pool.size() == obj_pool._objects.size());
}
