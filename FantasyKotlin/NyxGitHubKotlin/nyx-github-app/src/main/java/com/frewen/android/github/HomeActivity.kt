package com.frewen.android.github

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import androidx.appcompat.widget.Toolbar
import androidx.fragment.app.Fragment
import com.frewen.android.github.adapter.FragmentPagerViewAdapter
import dagger.android.AndroidInjector
import dagger.android.DispatchingAndroidInjector
import dagger.android.support.HasSupportFragmentInjector
import kotlinx.android.synthetic.main.activity_home.*
import javax.inject.Inject

class HomeActivity : AppCompatActivity(), HasSupportFragmentInjector, Toolbar.OnMenuItemClickListener {

    @Inject
    lateinit var dispatchingAndroidInjector: DispatchingAndroidInjector<Fragment>

    /**
     * fragment列表
     */
    @Inject
    lateinit var homeFragmentList: MutableList<Fragment>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_home)

        Log.i("HomeActivity", "onCreate")

        initViewPager()
    }

    private fun initViewPager() {
        home_view_pager.adapter = FragmentPagerViewAdapter(homeFragmentList, supportFragmentManager)
        home_navigation_tab_bar.setViewPager(home_view_pager, 0)
        // 缓存所有的ViewPager页面
        home_view_pager.offscreenPageLimit = homeFragmentList.size
    }

    /**
     * 返回注入的AndroidInjector对象
     */
    override fun supportFragmentInjector(): AndroidInjector<Fragment> {
        return dispatchingAndroidInjector
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.main_toolbar_menu, menu)
        return true
    }

    override fun onMenuItemClick(item: MenuItem?): Boolean {
        when (item?.itemId) {
            R.id.action_search -> {

            }
        }
        return true
    }


}
