#ifndef _ThreadPool_HPP
#define _ThreadPool_HPP

#include "vision/util/Runnable.h"
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

//!
//! convenience macro to log with file and line information
//!
#ifdef __SOLA_LOGGING_ENABLED
#define __SOLA_LOG(level, msg) sola::level(msg, __FILE__, __LINE__);
#else
#define __SOLA_LOG(level, msg)
#endif /* __SOLA_LOGGING_ENABLED */

namespace aura::vision {

class LoggerIface {
public:
    //! ctor
    LoggerIface(void) = default;

    //! dtor
    virtual ~LoggerIface(void) = default;

    //! copy ctor
    LoggerIface(const LoggerIface &) = default;

    //! assignment operator
    LoggerIface &operator=(const LoggerIface &) = default;

public:
    //!
    //! debug logging
    //!
    //! \param msg message to be logged
    //! \param file file from which the message is coming
    //! \param line line in the file of the message
    //!
    virtual void debug(const std::string &msg, const std::string &file, std::size_t line) = 0;

    //!
    //! info logging
    //!
    //! \param msg message to be logged
    //! \param file file from which the message is coming
    //! \param line line in the file of the message
    //!
    virtual void info(const std::string &msg, const std::string &file, std::size_t line) = 0;

    //!
    //! warn logging
    //!
    //! \param msg message to be logged
    //! \param file file from which the message is coming
    //! \param line line in the file of the message
    //!
    virtual void warn(const std::string &msg, const std::string &file, std::size_t line) = 0;

    //!
    //! error logging
    //!
    //! \param msg message to be logged
    //! \param file file from which the message is coming
    //! \param line line in the file of the message
    //!
    virtual void error(const std::string &msg, const std::string &file, std::size_t line) = 0;
};

//!
//! default logger class provided by the library
//!
class logger : public LoggerIface {
public:
    //!
    //! log level
    //!
    enum class log_level {
        error = 0,
        warn = 1,
        info = 2,
        debug = 3
    };

public:
    //! ctor
    logger(log_level level = log_level::info);

    //! dtor
    ~logger(void) = default;

    //! copy ctor
    logger(const logger &) = default;

    //! assignment operator
    logger &operator=(const logger &) = default;

public:
    //!
    //! debug logging
    //!
    //! \param msg message to be logged
    //! \param file file from which the message is coming
    //! \param line line in the file of the message
    //!
    void debug(const std::string &msg, const std::string &file, std::size_t line);

    //!
    //! info logging
    //!
    //! \param msg message to be logged
    //! \param file file from which the message is coming
    //! \param line line in the file of the message
    //!
    void info(const std::string &msg, const std::string &file, std::size_t line);

    //!
    //! warn logging
    //!
    //! \param msg message to be logged
    //! \param file file from which the message is coming
    //! \param line line in the file of the message
    //!
    void warn(const std::string &msg, const std::string &file, std::size_t line);

    //!
    //! error logging
    //!
    //! \param msg message to be logged
    //! \param file file from which the message is coming
    //! \param line line in the file of the message
    //!
    void error(const std::string &msg, const std::string &file, std::size_t line);

private:
    //!
    //! current log level in use
    //!
    log_level m_level;

    //!
    //! mutex used to serialize logs in multithreaded environment
    //!
    std::mutex m_mutex;
};

//!
//! variable containing the current logger
//! by default, not set (no logs)
//!
extern std::unique_ptr<LoggerIface> active_logger;

//!
//! debug logging
//! convenience function used internally to call the logger
//!
//! \param msg message to be logged
//! \param file file from which the message is coming
//! \param line line in the file of the message
//!
void debug(const std::string &msg, const std::string &file, std::size_t line);

//!
//! info logging
//! convenience function used internally to call the logger
//!
//! \param msg message to be logged
//! \param file file from which the message is coming
//! \param line line in the file of the message
//!
void info(const std::string &msg, const std::string &file, std::size_t line);

//!
//! warn logging
//! convenience function used internally to call the logger
//!
//! \param msg message to be logged
//! \param file file from which the message is coming
//! \param line line in the file of the message
//!
void warn(const std::string &msg, const std::string &file, std::size_t line);

//!
//! error logging
//! convenience function used internally to call the logger
//!
//! \param msg message to be logged
//! \param file file from which the message is coming
//! \param line line in the file of the message
//!
void error(const std::string &msg, const std::string &file, std::size_t line);


class ThreadPool {
public:
    ThreadPool(int initSize = 3);

    ~ThreadPool();

    bool isStarted() { return mIsStarted; }

    void start();

    void stop();

    void execute(Runnable *task);  //thread safe;

    static int mInitThreadsSize;

private:
    ThreadPool(const ThreadPool &);//禁止复制拷贝.
    const ThreadPool &operator=(const ThreadPool &);

    void threadLoop();

    Runnable *take();

    std::vector<std::thread *> mThreads;
    std::deque<Runnable *> mTasks;

    std::mutex mMutex;
    std::condition_variable mCond;
    bool mIsStarted;
};

} // namespace vision

#endif
